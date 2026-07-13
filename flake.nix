{
  description = "Development environment for DeepLung";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { nixpkgs, ... }:
    let
      systems = [
        "aarch64-darwin"
        "aarch64-linux"
        "x86_64-darwin"
        "x86_64-linux"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
    in
    {
      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs { inherit system; };
          python = pkgs.python3.withPackages (ps: with ps; [
            torch
            torchvision
            pandas
            numpy
            matplotlib
            scikit-learn
            tqdm
            pillow
            seaborn
            streamlit
            flask
            flask-cors
          ]);
        in
        {
          default = pkgs.mkShell {
            packages = [
              python
              pkgs.nodejs_22
            ];

            shellHook = ''
              export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
            '';
          };
        });
    };
}
