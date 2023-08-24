{
  description = "Haskell bindings for TensorFlow";

  inputs = {
    nixpkgs.url = "nixpkgs/nixpkgs-unstable";

    flake-utils.url = "github:numtide/flake-utils";

    flake-compat = {
      url = "github:edolstra/flake-compat";
      flake = false;
    };

    tensorflowSubModule = {
      url =
        "github:tensorflow/tensorflow/b36436b087bd8e8701ef51718179037cccdfc26e";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, flake-utils, tensorflowSubModule, ... }:
    flake-utils.lib.eachSystem [ "x86_64-linux" ] (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ (import ./nix/overlay.nix tensorflowSubModule) ];
        };

        tfPackages = [
          "tensorflow"
          "tensorflow-core-ops"
          "tensorflow-logging"
          "tensorflow-opgen"
          "tensorflow-ops"
          "tensorflow-proto"
          "tensorflow-records"
          "tensorflow-records-conduit"
          "tensorflow-test"
        ];

      in {
        packages = builtins.listToAttrs (builtins.map (p: {
          name = p;
          value = pkgs.haskellPackages."${p}";
        }) tfPackages);

        devShells = let
          devDeps = with pkgs; [
            cabal-install
            haskell-language-server
            hlint
            ormolu
          ];

          devPkgs = builtins.listToAttrs (builtins.map (p: {
            name = p;
            value = pkgs.mkShell {
              buildInputs = let
                pkgEnv = pkgs.haskellPackages.ghcWithPackages (ghcPkgs:
                  ghcPkgs."${p}".buildInputs
                  ++ ghcPkgs."${p}".propagatedBuildInputs
                  ++ ghcPkgs."${p}".propagatedNativeBuildInputs
                  ++ ghcPkgs."${p}".nativeBuildInputs ++ devDeps);
              in [ pkgEnv ];
            };
          }) tfPackages);

        in devPkgs // { default = devPkgs.tensorflow; };
      }) // {
        overlays.default = import ./nix/overlay.nix tensorflowSubModule;
      };
}
