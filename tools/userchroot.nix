#
# This script is provided by @grwlf (grwlf@gmail.com) and is supported on best
# effort basis.  TensorFlow development team does not regularly test this script
# and can't answer any questions about it.
#
{ pkgs ? import <nixpkgs> {} }:

with pkgs;

let
  tensorflow-c = stdenv.mkDerivation {

    name = "tensorflow-c";

    src = fetchurl {
      url = "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.7.0.tar.gz";
      sha256 = "621642b1fddd3831e048817d2220d9d7cf8ba359ac81c83a808bcdd9a982ee90";
    };

    buildCommand = ''
      . $stdenv/setup
      mkdir -pv $out
      tar -C $out -xzf $src
    '';
  };

  fhs = pkgs.buildFHSUserEnv {
    name = "fhs";
    targetPkgs = pkgs:
      with pkgs.python3Packages;
      with pkgs.haskellPackages;
      with pkgs;
      assert stack.version >= "1.4";
      assert ghc.version >= "8";
      [
        snappy
        snappy.dev
        protobuf3_2
        stack
        ghc
        iana-etc
        pkgs.zlib
        pkgs.zlib.dev
        tensorflow-c
        gcc
        gdb
      ];
    runScript = "bash";
    profile = ''
      export USE_CCACHE=1
      export LIBRARY_PATH=/usr/lib64
      export LD_LIBRARY_PATH=$(pwd):/usr/lib64
      export ACLOCAL_PATH=/usr/share/aclocal
      export LANG=C
      export LC_ALL=en_US.utf-8
      export PATH="$HOME/.local/bin:$PATH"
    '';
  };

in
stdenv.mkDerivation {
  name = fhs.name + "-env";
  nativeBuildInputs = [ fhs ];
  shellHook = "exec fhs";
}
