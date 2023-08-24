tensorflowSubModule: final: prev:

let
  /* The tensorflow submodule in third_party/tensorflow and the symlinks in the
     Haskell-package directories will always be removed by Nix.
     Thus construct a source here where the submodule is a normal directory
     whenever required.
  */
  tfSrc = prev.runCommand "prepTensorflowSrc" {
    src = prev.nix-gitignore.gitignoreSource [
      "nix/"
      "flake.nix"
      "flake.lock"
      "result"
    ] ./..;
  } ''
    cp -r $src/* .

    chmod -R +rwx ./third_party
    cp -r ${tensorflowSubModule} ./third_party/tensorflow

    chmod -R +rwx tensorflow
    rm -rf tensorflow/third_party
    cp -r ${tensorflowSubModule} ./tensorflow/third_party

    for d in tensorflow-opgen tensorflow-proto; do
      chmod +rwx $d $d/third_party
      rm -rf $d/third_party
      mkdir $d/third_party
      cp -r ${tensorflowSubModule} $d/third_party/tensorflow
    done
    cp -r . $out
  '';

  haskellOverrides = final: prev: hfinal: hprev: {
    tensorflow = hfinal.callCabal2nix "tensorflow" "${tfSrc}/tensorflow" { };

    tensorflow-core-ops =
      hfinal.callCabal2nix "tensorflow-core-ops" "${tfSrc}/tensorflow-core-ops"
      { };

    tensorflow-logging =
      hfinal.callCabal2nix "tensorflow-logging" "${tfSrc}/tensorflow-logging"
      { };

    tensorflow-opgen =
      hfinal.callCabal2nix "tensorflow-opgen" "${tfSrc}/tensorflow-opgen" { };

    tensorflow-ops =
      hfinal.callCabal2nix "tensorflow-ops" "${tfSrc}/tensorflow-ops" { };

    tensorflow-proto = let
      c2n =
        hfinal.callCabal2nix "tensorflow-proto" "${tfSrc}/tensorflow-proto" { };
    in prev.haskell.lib.overrideCabal c2n (drv: {
      libraryToolDepends = drv.libraryToolDepends ++ [ prev.protobuf ];
    });

    tensorflow-records =
      hfinal.callCabal2nix "tensorflow-records" "${tfSrc}/tensorflow-records"
      { };

    tensorflow-records-conduit =
      hfinal.callCabal2nix "tensorflow-records-conduit"
      "${tfSrc}/tensorflow-records-conduit" { };

    tensorflow-test =
      hfinal.callCabal2nix "tensorflow-test" "${tfSrc}/tensorflow-test" { };
  };

in {
  haskell = prev.haskell // {
    packages = prev.haskell.packages // (builtins.mapAttrs
      (key: val: val.override { overrides = haskellOverrides final prev; }) {
        inherit (prev.haskell.packages) ghc8107 ghc902 ghc923;
      });
  };
}
