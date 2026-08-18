//! Installation of the Woxi Jupyter kernelspec (`woxi install-kernel`).
//!
//! The kernelspec files are embedded in the binary, so the command works from
//! any working directory and from a binary installed via `cargo install` /
//! a release archive, which has no `kernelspec/` directory next to it.
//! They are materialised into a temporary staging directory and handed to
//! `jupyter kernelspec install`; when the `jupyter` CLI is not on `PATH` the
//! spec is written into Jupyter's kernels directory directly.

use std::io;
use std::path::{Path, PathBuf};
use std::{env, fs};

/// Name of the kernel in Jupyter, i.e. the kernelspec directory name.
const KERNEL_NAME: &str = "woxi";

/// The repository's `kernelspec/woxi` is the single source of truth for the
/// kernel's metadata and icons; it is compiled into the binary verbatim.
const KERNEL_JSON: &str = include_str!("../kernelspec/woxi/kernel.json");
const LOGO_32: &[u8] = include_bytes!("../kernelspec/woxi/logo-32x32.png");
const LOGO_64: &[u8] = include_bytes!("../kernelspec/woxi/logo-64x64.png");

/// Where the kernelspec is installed to.
#[derive(Clone, Copy)]
enum Scope {
  /// The current user's Jupyter data directory.
  User,
  /// The machine-wide Jupyter data directory.
  System,
}

/// Install Woxi as a Jupyter kernel.
///
/// `user` takes precedence over `system`, and a user installation is the
/// default when neither flag is given.
pub(crate) fn install(user: bool, system: bool) -> io::Result<()> {
  let scope = if system && !user {
    Scope::System
  } else {
    Scope::User
  };

  // Stage the spec in a temporary directory. `jupyter kernelspec install`
  // derives the kernel name from the directory name, hence the `woxi`
  // subdirectory.
  let staging_root =
    env::temp_dir().join(format!("woxi-kernelspec-{}", std::process::id()));
  let staging = staging_root.join(KERNEL_NAME);
  let _ = fs::remove_dir_all(&staging_root);
  let result =
    write_spec(&staging).and_then(|()| install_spec(&staging, scope));
  let _ = fs::remove_dir_all(&staging_root);
  result
}

/// Write `kernel.json` and both logos into `dir`, creating it if needed.
fn write_spec(dir: &Path) -> io::Result<()> {
  fs::create_dir_all(dir)?;
  fs::write(dir.join("kernel.json"), kernel_json())?;
  fs::write(dir.join("logo-32x32.png"), LOGO_32)?;
  fs::write(dir.join("logo-64x64.png"), LOGO_64)?;
  Ok(())
}

/// The `kernel.json` to install.
///
/// `argv[0]` is rewritten to the absolute path of the running executable so
/// the kernel also starts when the Jupyter server runs with a `PATH` that
/// does not contain `woxi` (a virtualenv, a systemd unit, JupyterHub …).
/// The embedded `woxi` is kept when the path cannot be determined.
fn kernel_json() -> String {
  let Some(exe) = current_exe() else {
    return KERNEL_JSON.to_string();
  };
  let Ok(mut spec) = serde_json::from_str::<serde_json::Value>(KERNEL_JSON)
  else {
    return KERNEL_JSON.to_string();
  };
  if let Some(argv0) = spec
    .get_mut("argv")
    .and_then(|argv| argv.as_array_mut())
    .and_then(|argv| argv.first_mut())
  {
    *argv0 = serde_json::Value::String(exe);
  }
  format!(
    "{}\n",
    serde_json::to_string_pretty(&spec).unwrap_or_default()
  )
}

/// The absolute path of the running `woxi` binary, if it is representable as
/// UTF-8 (`kernel.json` is JSON, so non-UTF-8 paths cannot be stored).
fn current_exe() -> Option<String> {
  let exe = env::current_exe().ok()?;
  // Resolves `./woxi` and symlinks; the un-canonicalised path is still
  // absolute and usable, so a failure here is not fatal.
  let exe = fs::canonicalize(&exe).unwrap_or(exe);
  exe.to_str().map(ToString::to_string)
}

/// Hand `spec_dir` to `jupyter kernelspec install`, falling back to a direct
/// copy into Jupyter's kernels directory when the `jupyter` CLI is missing.
fn install_spec(spec_dir: &Path, scope: Scope) -> io::Result<()> {
  let scope_flag = match scope {
    Scope::User => "--user",
    Scope::System => "--system",
  };
  let status = std::process::Command::new("jupyter")
    .args([
      "kernelspec",
      "install",
      "--replace",
      "--name",
      KERNEL_NAME,
      scope_flag,
    ])
    .arg(spec_dir)
    .status();

  match status {
    Ok(status) if status.success() => {
      report_success(None);
      Ok(())
    }
    Ok(status) => Err(io::Error::other(format!(
      "Failed to install kernel. Exit code: {status}"
    ))),
    Err(err) if err.kind() == io::ErrorKind::NotFound => {
      // No Jupyter CLI on `PATH` — install the spec ourselves rather than
      // failing, so the kernel is ready for whichever Jupyter the user runs.
      let destination = kernels_dir(scope)?.join(KERNEL_NAME);
      copy_spec(spec_dir, &destination)?;
      report_success(Some(&destination));
      Ok(())
    }
    Err(err) => Err(err),
  }
}

/// Replace `destination` with the staged kernelspec.
fn copy_spec(spec_dir: &Path, destination: &Path) -> io::Result<()> {
  if destination.exists() {
    fs::remove_dir_all(destination)?;
  }
  fs::create_dir_all(destination)?;
  for entry in fs::read_dir(spec_dir)? {
    let entry = entry?;
    fs::copy(entry.path(), destination.join(entry.file_name()))?;
  }
  Ok(())
}

fn report_success(destination: Option<&Path>) {
  match destination {
    Some(path) => {
      println!("Woxi kernel installed successfully in {}", path.display());
    }
    None => println!("Woxi kernel installed successfully!"),
  }
  println!(
    "You can now use it in Jupyter Lab or Notebook by selecting 'Woxi' from the kernel list."
  );
}

/// Jupyter's kernels directory for `scope`, mirroring the locations
/// `jupyter_core` uses.
fn kernels_dir(scope: Scope) -> io::Result<PathBuf> {
  Ok(jupyter_data_dir(scope)?.join("kernels"))
}

fn jupyter_data_dir(scope: Scope) -> io::Result<PathBuf> {
  if let Scope::System = scope {
    return Ok(if cfg!(windows) {
      PathBuf::from(non_empty_var("PROGRAMDATA").ok_or_else(|| {
        io::Error::other("Cannot locate %PROGRAMDATA%: set JUPYTER_DATA_DIR")
      })?)
      .join("jupyter")
    } else {
      PathBuf::from("/usr/local/share/jupyter")
    });
  }

  // An explicit data directory wins over the platform default.
  if let Some(dir) = non_empty_var("JUPYTER_DATA_DIR") {
    return Ok(PathBuf::from(dir));
  }

  if cfg!(windows) {
    return Ok(
      PathBuf::from(non_empty_var("APPDATA").ok_or_else(|| {
        io::Error::other("Cannot locate %APPDATA%: set JUPYTER_DATA_DIR")
      })?)
      .join("jupyter"),
    );
  }

  let home = dirs::home_dir().ok_or_else(|| {
    io::Error::other("Cannot locate the home directory: set JUPYTER_DATA_DIR")
  })?;
  if cfg!(target_os = "macos") {
    return Ok(home.join("Library").join("Jupyter"));
  }
  let data_home = non_empty_var("XDG_DATA_HOME")
    .map_or_else(|| home.join(".local").join("share"), PathBuf::from);
  Ok(data_home.join("jupyter"))
}

/// `env::var` treating an empty value as unset, like `jupyter_core` does.
fn non_empty_var(name: &str) -> Option<String> {
  env::var(name).ok().filter(|value| !value.is_empty())
}
