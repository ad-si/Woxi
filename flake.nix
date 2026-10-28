{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      utils,
    }:
    utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            bash
            cargo-insta
            cargo-nextest
            cargo-outdated
            cargo-sweep # `make sweep`
            coreutils # cat
            gnumake
            gnused
            lld
            nodejs
            prek
            # The Rust toolchain itself comes from rustup (pinned by
            # rust-toolchain.toml), never from nixpkgs: a nix cargo next to
            # the rustup proxies in ~/.cargo/bin lets PATH order pick the
            # compiler, and every switch rebuilds target/ from scratch.
            rustup
            uv
            wasm-pack
          ];
          # Use LLVM's lld for host builds on macOS — substantially faster
          # than the default ld64, and linking dominates incremental build
          # time for a crate this size. Kept out of .cargo/config.toml so
          # builds outside this dev shell (Homebrew, plain checkouts) fall
          # back to the system linker instead of failing on a missing lld.
          # Scoped to the host triple so wasm builds are unaffected.
          env = pkgs.lib.optionalAttrs pkgs.stdenv.hostPlatform.isDarwin {
            CARGO_TARGET_AARCH64_APPLE_DARWIN_RUSTFLAGS = "-C link-arg=-fuse-ld=lld";
          };
        };
        formatter = pkgs.nixfmt-tree; # Format this file with `nix fmt`
      }
    );
}
