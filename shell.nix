{ghc ? null }:

let
  # Use pinned packages
  _nixpkgs = import <nixpkgs> {};
  nixpkgs = _nixpkgs.fetchFromGitHub (_nixpkgs.lib.importJSON ./nix/src.json);
  pkgs = import nixpkgs {};

  # Either use specified GHC or use GHC 8.2.2 (which we need for LTS 11.9)
  myghc = if isNull ghc then pkgs.haskell.compiler.ghc822 else ghc;

  # Fetch tensorflow library
  tensorflow-c = pkgs.stdenv.mkDerivation {
    name = "tensorflow-c";
    src = pkgs.fetchurl {
      url = "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.8.0.tar.gz";
      sha256 = "0qzy15rc3x961cyi3bqnygrcnw4x69r28xkwhpwrv1r0gi6k73ha";
    };

    # Patch library to use our libc, libstdc++ and others
    buildCommand = ''
      . $stdenv/setup
      mkdir -pv $out
      tar -C $out -xzf $src
      chmod +w $out/lib/libtensorflow.so
      ${pkgs.patchelf}/bin/patchelf --set-rpath "${pkgs.stdenv.cc.libc}/lib:${pkgs.stdenv.cc.cc.lib}/lib" $out/lib/libtensorflow.so
      chmod -w $out/lib/libtensorflow.so
    '';
  };

  # Wrapped stack executable that uses the nix-provided GHC
  stack = pkgs.stdenv.mkDerivation {
      name = "stack-system-ghc";
      builder = pkgs.writeScript "stack" ''
        source $stdenv/setup
        mkdir -p $out/bin
        makeWrapper ${pkgs.stack}/bin/stack $out/bin/stack \
          --add-flags --system-ghc
      '';
      buildInputs = [ pkgs.makeWrapper ];
    };
in
  pkgs.haskell.lib.buildStackProject {
    ghc = myghc;
    stack = stack;
    name = "tf-env";
    buildInputs =
        [
          pkgs.snappy
          pkgs.zlib
          pkgs.protobuf3_3
          tensorflow-c
          stack
        ];
  }
