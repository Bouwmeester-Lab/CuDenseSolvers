vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO Bouwmeester-Lab/CuDenseSolvers
    REF master  # Or a tag like v0.1.0
    SHA512 0  # Placeholder, use actual hash below
)

# Install headers from CuDenseSolvers/ folder
file(INSTALL
    "${SOURCE_PATH}/CuDenseSolvers"
    DESTINATION "${CURRENT_PACKAGES_DIR}/include"
    FILES_MATCHING PATTERN "*.hpp"
)