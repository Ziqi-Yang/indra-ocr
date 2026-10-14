{
  description = "indra-ocr: Rust OCR pipeline and inference engine for Indra OCR";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" ];
      forAll = f: nixpkgs.lib.genAttrs systems (s: f nixpkgs.legacyPackages.${s});
    in {
      devShells = forAll (pkgs: {
        default = pkgs.mkShell {
          packages = with pkgs; [ cargo rustc rustfmt clippy rust-analyzer onnxruntime pdfium uv ];
          # Runtime libraries the crates load dynamically.
          ORT_DYLIB_PATH = "${pkgs.onnxruntime}/lib/libonnxruntime.so";
          PDFIUM_DYNAMIC_LIB_PATH = "${pkgs.pdfium}/lib";
        };
      });
    };
}
