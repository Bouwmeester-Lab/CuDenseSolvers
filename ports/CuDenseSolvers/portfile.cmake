vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO Bouwmeester-Lab/CuDenseSolvers
    REF v0.2  # Or a tag like v0.1.0
    SHA512 85dc5f0f008764cd7454d877449586295e8f868286025dc490007a2540f414d9aa294afebeb93b6237cdf309e3a5b50fa56eb475d8de8ca4b2e1b3843f3f24d0
)

# Install headers from CuDenseSolvers/include/
file(INSTALL
    "${SOURCE_PATH}/CuDenseSolvers/include/"
    DESTINATION "${CURRENT_PACKAGES_DIR}/include/CuDenseSolvers"
    FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hpp"
        PATTERN "*.cuh"
)

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE.txt")