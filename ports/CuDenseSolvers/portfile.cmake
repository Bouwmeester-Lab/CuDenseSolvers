vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO Bouwmeester-Lab/CuDenseSolvers
    REF v0.1  # Or a tag like v0.1.0
    SHA512 bf6eadfc7cb1bbf6176744c6398ee5b0ed1241c0d4b6a985c17d5f692380d62b38e54a9246155fecfb00f807e9b976be1adbdfc7b6cfe23e8a3350e2c1b20637
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