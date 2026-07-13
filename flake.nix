# {
#   description = "Development environment for DeepLung";
#
#   inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
#
#   outputs = { nixpkgs, ... }:
#     let
#       systems = [
#         "aarch64-darwin"
#         "aarch64-linux"
#         "x86_64-darwin"
#         "x86_64-linux"
#       ];
#       forAllSystems = nixpkgs.lib.genAttrs systems;
#     in
#     {
#       devShells = forAllSystems (system:
#         let
#           pkgs = import nixpkgs { inherit system; };
#           python = pkgs.python3.withPackages (ps: with ps; [
#             torch
#             torchvision
#             pandas
#             numpy
#             matplotlib
#             scikit-learn
#             tqdm
#             pillow
#             seaborn
#             streamlit
#             flask
#             flask-cors
#           ]);
#         in
#         {
#           default = pkgs.mkShell {
#             packages = [
#               python
#               pkgs.nodejs_22
#             ];
#
#             shellHook = ''
#               export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
#             '';
#           };
#         });
#     };
# }

{
  description = "DeepLung - dev environment + deployable packages";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs, ... }:
    let
      systems = [
        "aarch64-darwin"
        "aarch64-linux"
        "x86_64-darwin"
        "x86_64-linux"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;

      pythonPackagesFor = pkgs: ps: with ps; [
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
        gunicorn
      ];
    in
    {
      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs { inherit system; };
          python = pkgs.python3.withPackages (pythonPackagesFor pkgs);
        in
        {
          default = pkgs.mkShell {
            packages = [
              python
              pkgs.nodejs_22
              pkgs.pnpm
            ];
            shellHook = ''
              export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
            '';
          };
        });

      packages = forAllSystems (system:
        let
          pkgs = import nixpkgs { inherit system; };
          python = pkgs.python3.withPackages (pythonPackagesFor pkgs);
        in
        {
          # Backend: src/ + checkpoints/ ONLY (never data/ - ~10k training
          # images, must not enter the Nix store). Preserves the sibling
          # layout server.py's resolve_model_path() expects: checkpoints/
          # next to src/, not inside it.
          backend = pkgs.stdenv.mkDerivation {
            pname = "deeplung-backend";
            version = "0.1.0";

            src = pkgs.lib.fileset.toSource {
              root = ./.;
              fileset = pkgs.lib.fileset.unions [
                ./src
                ./checkpoints
              ];
            };

            nativeBuildInputs = [ pkgs.makeWrapper ];
            dontBuild = true;

            installPhase = ''
              mkdir -p $out/share/deeplung
              cp -r src $out/share/deeplung/src
              cp -r checkpoints $out/share/deeplung/checkpoints

              mkdir -p $out/bin
              makeWrapper ${python}/bin/gunicorn $out/bin/deeplung-backend \
                --add-flags "server:app --bind 127.0.0.1:8000" \
                --run "cd $out/share/deeplung/src"
            '';
          };

          frontend = pkgs.stdenv.mkDerivation (finalAttrs: {
            pname = "deeplung-frontend";
            version = "0.1.0";
            src = ./frontend;

            nativeBuildInputs = [ pkgs.nodejs_22 pkgs.pnpmConfigHook pkgs.pnpm_10 ];

            pnpmDeps = pkgs.fetchPnpmDeps {
              inherit (finalAttrs) pname version src;
              pnpm = pkgs.pnpm_10;
              fetcherVersion = 3;
              hash = "sha256-JfEphYaKwABcjRkG5bau4CKaCOvZEkbNt4xboRwR79k=";
            };

            buildPhase = ''
              pnpm run build
            '';

            installPhase = ''
              mkdir -p $out
              cp -r dist/* $out/
            '';
          });

          # frontend = pkgs.stdenv.mkDerivation (finalAttrs: {
          #   pname = "deeplung-frontend";
          #   version = "0.1.0";
          #   src = ./frontend;
          #
          #   nativeBuildInputs = [ pkgs.nodejs_22 pkgs.pnpm.configHook pkgs.pnpm_10 ];
          #
          #   pnpmDeps = pkgs.pnpm.fetchDeps {
          #     inherit (finalAttrs) pname version src;
          #     pnpm = pkgs.pnpm_10;
          #     fetcherVersion = 3;
          #     hash = ""; # nix will print the correct hash on first build; paste it in
          #   };
          #
          #   buildPhase = ''
          #     pnpm run build
          #   '';
          #
          #   installPhase = ''
          #     mkdir -p $out
          #     cp -r dist/* $out/
          #   '';
          # });

          default = self.packages.${system}.backend;
        });

      nixosModules.default = { config, lib, pkgs, ... }:
        let
          system = pkgs.system;
          backend = self.packages.${system}.backend;
          frontend = self.packages.${system}.frontend;
        in
        {
          systemd.services.deeplung-backend = {
            description = "DeepLung Flask backend";
            after = [ "network.target" ];
            wantedBy = [ "multi-user.target" ];
            serviceConfig = {
              ExecStart = "${backend}/bin/deeplung-backend";
              Restart = "on-failure";
              DynamicUser = true;
              MemoryMax = "4G";
            };
          };

          services.nginx.virtualHosts."deeplung.example.com" = {
            enableACME = true;
            forceSSL = true;
            root = "${frontend}";
            locations."/api/" = {
              proxyPass = "http://127.0.0.1:8000/";
            };
            locations."/" = {
              tryFiles = "$uri $uri/ /index.html";
            };
          };
        };
    };
}
