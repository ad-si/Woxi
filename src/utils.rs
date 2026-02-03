use rand::distributions::{Alphanumeric, Distribution};
use std::env;
use std::fs;
use std::path::PathBuf;

pub fn rand_str(length: usize) -> String {
  let rng = rand::thread_rng();
  let characters: Vec<char> = Alphanumeric
    .sample_iter(rng)
    .map(|c| c.into())
    .take(length)
    .collect();
  characters.iter().collect::<String>()
}

pub fn create_file(
  filename_opt: Option<String>,
) -> Result<PathBuf, std::io::Error> {
  let file_path = match filename_opt {
    Some(filename) => {
      let home_dir = env::current_dir().unwrap();
      home_dir.join(filename)
    }
    None => env::temp_dir().join(rand_str(16)),
  };

  fs::OpenOptions::new()
    .create_new(true)
    .write(true)
    .truncate(true)
    .open(&file_path)
    .map(|_| file_path)
}
