let pkgs = import <nixpkgs> {};
    deriv = pkgs.stdenv.mkDerivation rec {
      name = "barbaTOS";
      src = ./.;
    
      buildInputs = [
        pkgs.cmake
        pkgs.mpich
        pkgs.gtest
        pkgs.mkl
        pkgs.blas
        pkgs.cudatoolkit_11_2
        pkgs.cutensor_cudatoolkit_11_2
        pkgs.addOpenGLRunpath
      ];
    
      configurePhase = ''
        cmake . -DENABLE_GPU=ON -DCMAKE_BUILD_TYPE=Release
      '';
    
      buildPhase = ''
        make -j4
      '';
    
      installPhase = ''
        mkdir -p $out
        mv bin/* $out
      '';

      postFixup = ''
        addOpenGLRunpath $out/node_cli
      '';

    };
in deriv
