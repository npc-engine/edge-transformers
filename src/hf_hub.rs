use std::cmp::min;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::{env, fs, io};

use dirs::home_dir;
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use itertools::Itertools;
use reqwest::Client;
use sha2::{Digest, Sha256};

use crate::error::Error;

/// Builds huggingface URL to file from model id, revision and filename.
pub fn build_url_hf(model_id: &str, file_name: &str, revision: &str) -> String {
    format!(
        "https://huggingface.co/{}/resolve/{}/{}",
        model_id, revision, file_name
    )
}

/// Checks if file is present in cache, if not downloads a file from a URL and saves it to a local cache path.
/// Returns the path to the cached file.
pub fn hf_hub_download(
    model_id: &str,
    file_name: &str,
    revision: Option<&str>,
    cache_dir: Option<&str>,
) -> Result<PathBuf, Error> {
    let client = Client::new();
    let revision = if let Some(revision) = revision {
        revision
    } else {
        "main"
    };
    let file_url = build_url_hf(model_id, file_name, revision);
    let cache_dir = if let Some(cache_dir) = cache_dir {
        PathBuf::from(cache_dir)
    } else {
        get_cache_dir()
    };
    if !cache_dir.exists() {
        fs::create_dir_all(&cache_dir)?;
    }
    let cached = get_cached_file(file_url.as_str(), cache_dir.as_path());
    println!("Cached file: {:?}", cached);
    let tokyo_rt = tokio::runtime::Runtime::new().unwrap();
    if let Some(cached) = cached {
        return Ok(cached);
    } else {
        Ok(tokyo_rt.block_on(download_file(
            &client,
            file_url.as_str(),
            cache_dir.to_str().unwrap(),
        ))?)
    }
}

/// Downloads a file from a URL and saves it to a local cache path.
pub async fn download_file(client: &Client, url: &str, path: &str) -> Result<PathBuf, String> {
    println!("Downloading: {}", url);
    let res = client
        .get(url)
        .send()
        .await
        .or(Err(format!("Failed to GET from '{}'", &url)))?;
    if !res.status().is_success() {
        return Err(format!(
            "Failed to GET from '{}', status: {}",
            &url,
            res.status()
        ));
    }
    let total_size = res
        .content_length()
        .ok_or(format!("Failed to get content length from '{}'", &url))?;

    let pb = ProgressBar::new(total_size);
    pb.set_style(ProgressStyle::default_bar()
        .template("{msg}\n{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})").or(Err("Failed to set style"))?
        .progress_chars("#>-"));
    pb.set_message(format!("Downloading {}", url));

    let filename = url_to_filename(url);
    let path = Path::new(path).join(filename);

    let mut file = File::create(path.clone()).or(Err(format!(
        "Failed to create file '{}'",
        path.to_str().unwrap()
    )))?;
    let mut downloaded: u64 = 0;
    let mut stream = res.bytes_stream();

    while let Some(item) = stream.next().await {
        let chunk = item.or(Err(format!("Error while downloading file")))?;
        file.write_all(&chunk)
            .or(Err(format!("Error while writing to file")))?;
        let new = min(downloaded + (chunk.len() as u64), total_size);
        downloaded = new;
        pb.set_position(new);
    }
    pb.finish_with_message(format!("Downloaded {} to {}", url, path.to_str().unwrap()));
    return Ok(path);
}

/// Returns huggingface cache directory with edge-transformers subfolder.
pub fn get_cache_dir() -> PathBuf {
    let default_home = home_dir().unwrap().join(".cache");
    let hf_home =
        env::var("HF_HOME").unwrap_or_else(|_| default_home.to_str().unwrap().to_string());
    let xdg_home = env::var("XDG_CACHE_HOME").unwrap_or_else(|_| hf_home);
    let hf_cache_home = Path::new(&xdg_home).join("huggingface");

    let default_cache_path = hf_cache_home.join("edge-transformers");

    let hf_hub_cache = env::var("EDGE_TRANSFORMERS_HUB_CACHE")
        .unwrap_or_else(|_| default_cache_path.to_str().unwrap().to_string());

    PathBuf::from(hf_hub_cache)
}

pub fn get_ordered_labels_from_config(config_path: &str) -> Result<Vec<String>, Error> {
    let config = fs::read_to_string(config_path)?;
    let labels = serde_json::from_str::<serde_json::Value>(&config)?
        .get("id2label")
        .ok_or(Error::MissingId2Label)?
        .as_object()
        .ok_or(Error::MissingId2Label)?
        .iter()
        .map(|(k, v)| (k.to_string(), v.as_i64()))
        .sorted_by(|(_, a), (_, b)| a.cmp(b))
        .map(|(k, _)| k)
        .collect::<Vec<String>>();
    Ok(labels)
}

/// Returns the path to the cached file if it exists, otherwise returns None.
fn get_cached_file(url: &str, cache_dir: &Path) -> Option<PathBuf> {
    // Generate a filename for the file based on the URL
    let filename = url_to_filename(url);

    // Check if the file exists in the cache
    let file_path = cache_dir.join(filename);
    if file_path.exists() {
        Some(file_path)
    } else {
        None
    }
}

/// Converts a URL to a SHA256 hash filename.
fn url_to_filename(url: &str) -> PathBuf {
    let mut hasher = Sha256::default();
    hasher.update(url.as_bytes());
    let filename = hasher.finalize();
    let bytes = &filename[..];
    let filename = hex::encode(bytes);
    let filename = if url.ends_with(".h5") {
        format!("{}.h5", filename)
    } else {
        filename
    };
    PathBuf::from(filename)
}

#[cfg(test)]
mod tests {
    use reqwest::blocking::{get, Client};

    use super::*;

    #[test]
    fn test_get_cache_dir() {
        let cache_dir = get_cache_dir();
        // Try create
        fs::create_dir_all(cache_dir.clone()).unwrap();
        assert_eq!(
            cache_dir,
            home_dir()
                .unwrap()
                .join(".cache")
                .join("huggingface")
                .join("edge-transformers")
        );
    }

    #[test]
    fn test_url_to_filename() {
        let url = "https://huggingface.co/optimum/gpt2/raw/main/config.json";
        let filename = url_to_filename(url);
        println!("{}", filename.to_str().unwrap())
    }

    #[test]
    fn test_hf_download() {
        let model_id = "optimum/gpt2";
        let file_name = "decoder_with_past_model.onnx";
        let revision = "main";
        let path = hf_hub_download(model_id, file_name, Some(revision), Some("tmp_cache")).unwrap();
        // get file content from url
        let client = Client::new();
        let file_url = build_url_hf(model_id, file_name, revision);
        let res = client.get(file_url.as_str()).send().unwrap();
        let content = res.bytes().unwrap();
        // get downloaded file content
        let downloaded_content = fs::read(path).unwrap();
        assert!(get_cached_file(file_url.as_str(), Path::new("tmp_cache")).is_some());
        // Clean up
        // fs::remove_dir_all("tmp_cache").unwrap();
        assert_eq!(content.to_vec(), downloaded_content);
    }
}
