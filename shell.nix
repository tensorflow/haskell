{ghc ? null }:

let
  # Use pinned packages
  nixpkgs = with (builtins.fromJSON (builtins.readFile ./nix/src.json));
    builtins.fetchTarball {
      url = "https://github.com/${owner}/${repo}/archive/${rev}.tar.gz";
      inherit sha256;
    };

  pkgs = import nixpkgs {};
in
  pkgs.haskell.lib.buildStackProject {
    # Either use specified GHC or use GHC 8.6.5 (which we need for LTS 14.4)
    ghc = if isNull ghc then pkgs.haskell.compiler.ghc865 else ghc;
    extraArgs = "--system-ghc";
    name = "tf-env";
    buildInputs = with pkgs; [ snappy zlib protobuf libtensorflow ];
  }
