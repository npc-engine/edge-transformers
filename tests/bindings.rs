use interoptopus::util::NamespaceMappings;
use interoptopus::{Error, Interop};

#[test]
#[cfg_attr(miri, ignore)]
fn bindings_csharp() -> Result<(), Error> {
    use interoptopus_backend_csharp::{Config, Generator};

    let inventory = transformers_onnx_pipelines::ffi::ffi_inventory();
    let config = Config {
        class: "InteropClass".to_string(),
        dll_name: "transformers_onnx_pipelines".to_string(),
        namespace_mappings: NamespaceMappings::new("NPCEngine"),
        ..Config::default()
    };

    Generator::new(config, inventory).write_file("Interop.cs")?;

    Ok(())
}