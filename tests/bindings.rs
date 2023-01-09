use interoptopus::util::NamespaceMappings;
use interoptopus::{Error, Interop};

#[test]
#[cfg(feature = "generate-c")]
fn generate_c() -> Result<(), Error> {
    use interoptopus_backend_c::{Config, Generator};

    Generator::new(
        Config {
            ifndef: "edge_transformers".to_string(),
            ..Config::default()
        },
        edge_transformers::ffi::ffi_inventory(),
    )
    .write_file("c/edge_transformers.h")?;

    Ok(())
}

#[test]
#[cfg(feature = "generate-csharp")]
fn generate_csharp() -> Result<(), Error> {
    use interoptopus_backend_csharp::{Config, Generator};

    let inventory = edge_transformers::ffi::ffi_inventory();
    let config = Config {
        class: "Interop".to_string(),
        dll_name: "edge_transformers".to_string(),
        namespace_mappings: NamespaceMappings::new("EdgeTransformers"),
        ..Config::default()
    };

    Generator::new(config, inventory).write_file("csharp/EdgeTransformers.cs")?;
    Ok(())
}
