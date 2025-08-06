vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO Bouwmeester-Lab/CuDenseSolvers
    REF v0.1  # Or a tag like v0.1.0
    SHA512 12a26ade3310fa04dcdabe5a311a179aa321aef42bb92e5475f771eb1f8a250bbb419502d2b3e6bdc8a6b1bd84bd2ec9074f6637e0c7bb45facb26d5e4e3c583
)

# Install headers from CuDenseSolvers/include/
file(INSTALL
    "${SOURCE_PATH}/CuDenseSolvers/include"
    DESTINATION "${CURRENT_PACKAGES_DIR}"
    FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hpp"
        PATTERN "*.cuh"
)

vcpkg_install_copyright("${SOURCE_PATH}/LICENSE.txt")