# Install script for directory: /home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "TRUE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/home/ehsan/.espressif/tools/xtensa-esp32-elf/esp-2022r1-11.2.0/xtensa-esp32-elf/bin/xtensa-esp32-elf-objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/mbedtls" TYPE FILE PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ FILES
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/aes.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/aria.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/asn1.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/asn1write.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/base64.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/bignum.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/build_info.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/camellia.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/ccm.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/chacha20.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/chachapoly.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/check_config.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/cipher.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/cmac.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/compat-2.x.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/config_psa.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/constant_time.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/ctr_drbg.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/debug.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/des.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/dhm.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/ecdh.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/ecdsa.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/ecjpake.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/ecp.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/entropy.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/error.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/gcm.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/hkdf.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/hmac_drbg.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/mbedtls_config.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/md.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/md5.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/memory_buffer_alloc.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/net_sockets.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/nist_kw.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/oid.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/pem.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/pk.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/pkcs12.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/pkcs5.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/platform.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/platform_time.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/platform_util.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/poly1305.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/private_access.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/psa_util.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/ripemd160.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/rsa.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/sha1.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/sha256.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/sha512.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/ssl.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/ssl_cache.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/ssl_ciphersuites.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/ssl_cookie.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/ssl_ticket.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/threading.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/timing.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/version.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/x509.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/x509_crl.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/x509_crt.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/mbedtls/x509_csr.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/psa" TYPE FILE PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ FILES
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/psa/crypto.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/psa/crypto_builtin_composites.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/psa/crypto_builtin_primitives.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/psa/crypto_compat.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/psa/crypto_config.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/psa/crypto_driver_common.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/psa/crypto_driver_contexts_composites.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/psa/crypto_driver_contexts_primitives.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/psa/crypto_extra.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/psa/crypto_platform.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/psa/crypto_se_driver.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/psa/crypto_sizes.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/psa/crypto_struct.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/psa/crypto_types.h"
    "/home/ehsan/esp/esp-idf-v5.0/components/mbedtls/mbedtls/include/psa/crypto_values.h"
    )
endif()

