{
  description = "Image surprise detection using Duhem's Law and entropy";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python3;
        pythonEnv = python.withPackages (ps: with ps; [
          numpy
          pillow
          scipy
          matplotlib
          scikit-image
          scikit-learn
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pythonEnv
            pkgs.python3Packages.pip
            pkgs.python3Packages.virtualenv
          ];
          
          shellHook = ''
            # Create virtual environment if it doesn't exist
            if [ ! -d "venv" ]; then
              virtualenv venv
            fi
            source venv/bin/activate
            
            # Export useful variables
            export PIP_PREFIX="$(pwd)/_build/pip_packages"
            export PYTHONPATH="$(pwd)/_build/pip_packages/lib/python3.10/site-packages:$PYTHONPATH"
            export PATH="$(pwd)/_build/pip_packages/bin:$PATH"
            
            # Create directories if they don't exist
            mkdir -p _build/pip_packages
          '';
        };

        # For direct execution of the script
        apps.default = {
          type = "app";
          program = toString (pkgs.writers.writePython3 "surprise-detector" {
            libraries = with pkgs.python3Packages; [
              numpy
              pillow
              scipy
              matplotlib
              scikit-image
              scikit-learn
            ];
          } (builtins.readFile ./surprise_detector.py));
        };
      }
    );
}
