vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO Bouwmeester-Lab/CuDenseSolvers
    REF master  # Or a tag like v0.1.0
    SHA512 f32e25750c4cdfa880319558604ef7e1ced89cc505acad75a5f1624796cdf6873e835ac125bc53dd0af5622f7a7cfd388dff29e566c8efb12d739fbc93d865a8
)

# Install headers from CuDenseSolvers/include/
file(INSTALL
    "${SOURCE_PATH}/CuDenseSolvers/include"
    DESTINATION "${CURRENT_PACKAGES_DIR}"
    FILES_MATCHING PATTERN "*.hpp"
)