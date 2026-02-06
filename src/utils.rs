use rand::distributions::{Alphanumeric, Distribution};

pub fn rand_str(length: usize) -> String {
  let rng = rand::thread_rng();
  let characters: Vec<char> = Alphanumeric
    .sample_iter(rng)
    .map(|c| c.into())
    .take(length)
    .collect();
  characters.iter().collect::<String>()
}

#[cfg(not(target_arch = "wasm32"))]
pub fn create_file(
  filename_opt: Option<String>,
) -> Result<std::path::PathBuf, std::io::Error> {
  let file_path = match filename_opt {
    Some(filename) => {
      let home_dir = std::env::current_dir().unwrap();
      home_dir.join(filename)
    }
    None => std::env::temp_dir().join(rand_str(16)),
  };

  std::fs::OpenOptions::new()
    .create_new(true)
    .write(true)
    .truncate(true)
    .open(&file_path)
    .map(|_| file_path)
}
