#define _GNU_SOURCE
#include <ctype.h>
#include <dlfcn.h>
#include <elf.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

typedef int CUresult;
typedef void *CUmodule;
typedef void *CUlibrary;
typedef void *CUlinkState;
typedef int CUjitInputType;
typedef int CUjit_option;
typedef int CUlibraryOption;
typedef unsigned long long cuuint64_t;
typedef int CUdriverProcAddressQueryResult;

typedef int nvJitLinkResult;
typedef struct nvJitLink *nvJitLinkHandle;
typedef int nvJitLinkInputType;

typedef CUresult (*cuModuleLoadData_fn)(CUmodule *module, const void *image);
typedef CUresult (*cuModuleLoadDataEx_fn)(CUmodule *module, const void *image, unsigned int numOptions, CUjit_option *options, void **optionValues);
typedef CUresult (*cuModuleLoadFatBinary_fn)(CUmodule *module, const void *fatCubin);
typedef CUresult (*cuLibraryLoadData_fn)(CUlibrary *library, const void *code, CUjit_option *jitOptions, void **jitOptionsValues, unsigned int numJitOptions, CUlibraryOption *libraryOptions, void **libraryOptionValues, unsigned int numLibraryOptions);
typedef CUresult (*cuLibraryLoadFromFile_fn)(CUlibrary *library, const char *fileName, CUjit_option *jitOptions, void **jitOptionsValues, unsigned int numJitOptions, CUlibraryOption *libraryOptions, void **libraryOptionValues, unsigned int numLibraryOptions);
typedef CUresult (*cuLinkAddData_fn)(CUlinkState state, CUjitInputType type, void *data, size_t size, const char *name, unsigned int numOptions, CUjit_option *options, void **optionValues);
typedef CUresult (*cuLinkComplete_fn)(CUlinkState state, void **cubinOut, size_t *sizeOut);
typedef CUresult (*cuGetProcAddress_v2_fn)(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags, CUdriverProcAddressQueryResult *symbolStatus);
typedef CUresult (*cuGetProcAddress_legacy_fn)(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags);

typedef nvJitLinkResult (*nvJitLinkCreate_fn)(nvJitLinkHandle *handle, uint32_t numOptions, const char **options);
typedef nvJitLinkResult (*nvJitLinkAddData_fn)(nvJitLinkHandle handle, nvJitLinkInputType inputType, const void *data, size_t size, const char *name);
typedef nvJitLinkResult (*nvJitLinkComplete_fn)(nvJitLinkHandle handle);
typedef nvJitLinkResult (*nvJitLinkGetLinkedCubinSize_fn)(nvJitLinkHandle handle, size_t *size);
typedef nvJitLinkResult (*nvJitLinkGetLinkedCubin_fn)(nvJitLinkHandle handle, void *cubin);
typedef void *(*dlsym_fn)(void *handle, const char *symbol);
typedef void *(*dlvsym_fn)(void *handle, const char *symbol, const char *version);

static cuModuleLoadData_fn real_cuModuleLoadData;
static cuModuleLoadDataEx_fn real_cuModuleLoadDataEx;
static cuModuleLoadFatBinary_fn real_cuModuleLoadFatBinary;
static cuLibraryLoadData_fn real_cuLibraryLoadData;
static cuLibraryLoadFromFile_fn real_cuLibraryLoadFromFile;
static cuLinkAddData_fn real_cuLinkAddData;
static cuLinkAddData_fn real_cuLinkAddData_v2;
static cuLinkComplete_fn real_cuLinkComplete;
static cuGetProcAddress_v2_fn real_cuGetProcAddress_v2;
static cuGetProcAddress_legacy_fn real_cuGetProcAddress_legacy;

static nvJitLinkCreate_fn real_nvJitLinkCreate;
static nvJitLinkAddData_fn real_nvJitLinkAddData;
static nvJitLinkComplete_fn real_nvJitLinkComplete;
static nvJitLinkGetLinkedCubinSize_fn real_nvJitLinkGetLinkedCubinSize;
static nvJitLinkGetLinkedCubin_fn real_nvJitLinkGetLinkedCubin;
static nvJitLinkCreate_fn real___nvJitLinkCreate_12_8;
static nvJitLinkAddData_fn real___nvJitLinkAddData_12_8;
static nvJitLinkComplete_fn real___nvJitLinkComplete_12_8;
static nvJitLinkGetLinkedCubinSize_fn real___nvJitLinkGetLinkedCubinSize_12_8;
static nvJitLinkGetLinkedCubin_fn real___nvJitLinkGetLinkedCubin_12_8;
static dlsym_fn real_dlsym_fn;

CUresult cuModuleLoadData(CUmodule *module, const void *image);
CUresult cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int numOptions, CUjit_option *options, void **optionValues);
CUresult cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin);
CUresult cuLibraryLoadData(CUlibrary *library, const void *code, CUjit_option *jitOptions, void **jitOptionsValues, unsigned int numJitOptions, CUlibraryOption *libraryOptions, void **libraryOptionValues, unsigned int numLibraryOptions);
CUresult cuLibraryLoadFromFile(CUlibrary *library, const char *fileName, CUjit_option *jitOptions, void **jitOptionsValues, unsigned int numJitOptions, CUlibraryOption *libraryOptions, void **libraryOptionValues, unsigned int numLibraryOptions);
CUresult cuLinkAddData(CUlinkState state, CUjitInputType type, void *data, size_t size, const char *name, unsigned int numOptions, CUjit_option *options, void **optionValues);
CUresult cuLinkAddData_v2(CUlinkState state, CUjitInputType type, void *data, size_t size, const char *name, unsigned int numOptions, CUjit_option *options, void **optionValues);
CUresult cuLinkComplete(CUlinkState state, void **cubinOut, size_t *sizeOut);
CUresult cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags, CUdriverProcAddressQueryResult *symbolStatus);
CUresult cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags);
nvJitLinkResult nvJitLinkCreate(nvJitLinkHandle *handle, uint32_t numOptions, const char **options);
nvJitLinkResult nvJitLinkAddData(nvJitLinkHandle handle, nvJitLinkInputType inputType, const void *data, size_t size, const char *name);
nvJitLinkResult nvJitLinkComplete(nvJitLinkHandle handle);
nvJitLinkResult nvJitLinkGetLinkedCubinSize(nvJitLinkHandle handle, size_t *size);
nvJitLinkResult nvJitLinkGetLinkedCubin(nvJitLinkHandle handle, void *cubin);
nvJitLinkResult __nvJitLinkCreate_12_8(nvJitLinkHandle *handle, uint32_t numOptions, const char **options);
nvJitLinkResult __nvJitLinkAddData_12_8(nvJitLinkHandle handle, nvJitLinkInputType inputType, const void *data, size_t size, const char *name);
nvJitLinkResult __nvJitLinkComplete_12_8(nvJitLinkHandle handle);
nvJitLinkResult __nvJitLinkGetLinkedCubinSize_12_8(nvJitLinkHandle handle, size_t *size);
nvJitLinkResult __nvJitLinkGetLinkedCubin_12_8(nvJitLinkHandle handle, void *cubin);
void *dlsym(void *handle, const char *symbol);

static long next_id(void) {
    static volatile long counter = 0;
    return __sync_add_and_fetch(&counter, 1);
}

static const char *dump_dir(void) {
    const char *dir = getenv("CUDA_MODULE_DUMP_DIR");
    return (dir && dir[0]) ? dir : "asm/cublaslt/runtime_modules";
}

static void ensure_dir(void) {
    const char *dir = dump_dir();
    if (mkdir(dir, 0775) != 0 && errno != EEXIST) {
        fprintf(stderr, "[dump_cuda_modules] mkdir %s failed: %s\n", dir, strerror(errno));
    }
}

static void sanitize(char *dst, size_t dst_size, const char *src) {
    size_t out = 0;
    if (!src || !src[0]) {
        src = "anon";
    }
    for (size_t i = 0; src[i] && out + 1 < dst_size; ++i) {
        unsigned char c = (unsigned char)src[i];
        dst[out++] = (isalnum(c) || c == '_' || c == '-' || c == '.') ? (char)c : '_';
    }
    if (out == 0 && dst_size > 1) {
        dst[out++] = 'x';
    }
    dst[out] = '\0';
}

static int is_elf_image(const void *data) {
    const unsigned char *p = (const unsigned char *)data;
    return p && p[0] == 0x7f && p[1] == 'E' && p[2] == 'L' && p[3] == 'F';
}

static int looks_like_ptx(const void *data) {
    const char *p = (const char *)data;
    return p && (strncmp(p, ".version", 8) == 0 || strncmp(p, "//", 2) == 0 || strstr(p, "\n.version") != NULL);
}

static int looks_like_fatbin(const void *data) {
    const uint32_t *p = (const uint32_t *)data;
    return p && (p[0] == 0x466243b1u || p[0] == 0xba55ed50u || p[0] == 0x00100001u);
}

static const char *blob_ext(const void *data) {
    if (is_elf_image(data)) return "cubin";
    if (looks_like_ptx(data)) return "ptx";
    if (looks_like_fatbin(data)) return "fatbin";
    return "bin";
}

static size_t infer_elf64_size(const void *data) {
    const Elf64_Ehdr *eh = (const Elf64_Ehdr *)data;
    if (!is_elf_image(data) || eh->e_ident[EI_CLASS] != ELFCLASS64) {
        return 0;
    }
    if (eh->e_ehsize != sizeof(Elf64_Ehdr) || eh->e_shentsize != sizeof(Elf64_Shdr)) {
        return 0;
    }
    if (eh->e_shoff == 0 || eh->e_shnum == 0 || eh->e_shnum > 8192) {
        return 0;
    }
    size_t end = eh->e_ehsize;
    if (eh->e_phoff && eh->e_phnum && eh->e_phentsize) {
        size_t ph_end = (size_t)eh->e_phoff + (size_t)eh->e_phnum * (size_t)eh->e_phentsize;
        if (ph_end > end) end = ph_end;
    }
    size_t sh_end = (size_t)eh->e_shoff + (size_t)eh->e_shnum * (size_t)eh->e_shentsize;
    if (sh_end > end) end = sh_end;
    const Elf64_Shdr *sh = (const Elf64_Shdr *)((const char *)data + eh->e_shoff);
    for (uint16_t i = 0; i < eh->e_shnum; ++i) {
        if (sh[i].sh_type == SHT_NOBITS) continue;
        size_t sec_end = (size_t)sh[i].sh_offset + (size_t)sh[i].sh_size;
        if (sec_end > end) end = sec_end;
    }
    return (end > 0 && end < (size_t)1024 * 1024 * 1024) ? end : 0;
}

static size_t infer_blob_size(const void *data) {
    if (!data) return 0;
    if (is_elf_image(data)) {
        return infer_elf64_size(data);
    }
    if (looks_like_ptx(data)) {
        return strnlen((const char *)data, (size_t)128 * 1024 * 1024) + 1;
    }
    return 0;
}

static void dump_blob(const char *source, const char *label, const void *data, size_t size) {
    if (!data || size == 0) {
        fprintf(stderr, "[dump_cuda_modules] skip %s label=%s size=%zu\n", source, label ? label : "(null)", size);
        return;
    }
    if (size > (size_t)1024 * 1024 * 1024) {
        fprintf(stderr, "[dump_cuda_modules] skip huge blob %s label=%s size=%zu\n", source, label ? label : "(null)", size);
        return;
    }
    ensure_dir();
    char safe_source[96];
    char safe_label[192];
    sanitize(safe_source, sizeof(safe_source), source);
    sanitize(safe_label, sizeof(safe_label), label);
    char path[PATH_MAX];
    snprintf(path, sizeof(path), "%s/%04ld_%s_%s.%s", dump_dir(), next_id(), safe_source, safe_label, blob_ext(data));
    int fd = open(path, O_CREAT | O_TRUNC | O_WRONLY, 0664);
    if (fd < 0) {
        fprintf(stderr, "[dump_cuda_modules] open %s failed: %s\n", path, strerror(errno));
        return;
    }
    const char *p = (const char *)data;
    size_t remaining = size;
    while (remaining > 0) {
        ssize_t n = write(fd, p, remaining);
        if (n <= 0) {
            fprintf(stderr, "[dump_cuda_modules] write %s failed: %s\n", path, strerror(errno));
            break;
        }
        p += n;
        remaining -= (size_t)n;
    }
    close(fd);
    fprintf(stderr, "[dump_cuda_modules] wrote %s size=%zu source=%s label=%s\n", path, size, source, label ? label : "(null)");
}

static void dump_inferred(const char *source, const char *label, const void *data) {
    size_t size = infer_blob_size(data);
    dump_blob(source, label, data, size);
}

static dlsym_fn get_real_dlsym(void) {
    if (!real_dlsym_fn) {
        dlvsym_fn real_dlvsym = (dlvsym_fn)dlvsym(RTLD_NEXT, "dlvsym", "GLIBC_2.2.5");
        if (real_dlvsym) {
            real_dlsym_fn = (dlsym_fn)real_dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
        }
    }
    return real_dlsym_fn;
}

static void *load_next(const char *name) {
    dlsym_fn real_dlsym = get_real_dlsym();
    void *fn = real_dlsym ? real_dlsym(RTLD_NEXT, name) : NULL;
    if (!fn) {
        fprintf(stderr, "[dump_cuda_modules] dlsym(%s) failed: %s\n", name, dlerror());
    }
    return fn;
}

static void *wrapped_symbol(const char *symbol, void *pfn) {
    if (!symbol || !pfn) return pfn;
    if (strcmp(symbol, "cuModuleLoadData") == 0) {
        real_cuModuleLoadData = (cuModuleLoadData_fn)pfn;
        return (void *)cuModuleLoadData;
    } else if (strcmp(symbol, "cuModuleLoadDataEx") == 0) {
        real_cuModuleLoadDataEx = (cuModuleLoadDataEx_fn)pfn;
        return (void *)cuModuleLoadDataEx;
    } else if (strcmp(symbol, "cuModuleLoadFatBinary") == 0) {
        real_cuModuleLoadFatBinary = (cuModuleLoadFatBinary_fn)pfn;
        return (void *)cuModuleLoadFatBinary;
    } else if (strcmp(symbol, "cuLibraryLoadData") == 0) {
        real_cuLibraryLoadData = (cuLibraryLoadData_fn)pfn;
        return (void *)cuLibraryLoadData;
    } else if (strcmp(symbol, "cuLibraryLoadFromFile") == 0) {
        real_cuLibraryLoadFromFile = (cuLibraryLoadFromFile_fn)pfn;
        return (void *)cuLibraryLoadFromFile;
    } else if (strcmp(symbol, "cuLinkAddData") == 0) {
        real_cuLinkAddData = (cuLinkAddData_fn)pfn;
        return (void *)cuLinkAddData;
    } else if (strcmp(symbol, "cuLinkAddData_v2") == 0) {
        real_cuLinkAddData_v2 = (cuLinkAddData_fn)pfn;
        return (void *)cuLinkAddData_v2;
    } else if (strcmp(symbol, "cuLinkComplete") == 0) {
        real_cuLinkComplete = (cuLinkComplete_fn)pfn;
        return (void *)cuLinkComplete;
    } else if (strcmp(symbol, "cuGetProcAddress_v2") == 0) {
        real_cuGetProcAddress_v2 = (cuGetProcAddress_v2_fn)pfn;
        return (void *)cuGetProcAddress_v2;
    } else if (strcmp(symbol, "cuGetProcAddress") == 0) {
        real_cuGetProcAddress_legacy = (cuGetProcAddress_legacy_fn)pfn;
        return (void *)cuGetProcAddress;
    } else if (strcmp(symbol, "nvJitLinkCreate") == 0) {
        real_nvJitLinkCreate = (nvJitLinkCreate_fn)pfn;
        return (void *)nvJitLinkCreate;
    } else if (strcmp(symbol, "nvJitLinkAddData") == 0) {
        real_nvJitLinkAddData = (nvJitLinkAddData_fn)pfn;
        return (void *)nvJitLinkAddData;
    } else if (strcmp(symbol, "nvJitLinkComplete") == 0) {
        real_nvJitLinkComplete = (nvJitLinkComplete_fn)pfn;
        return (void *)nvJitLinkComplete;
    } else if (strcmp(symbol, "nvJitLinkGetLinkedCubinSize") == 0) {
        real_nvJitLinkGetLinkedCubinSize = (nvJitLinkGetLinkedCubinSize_fn)pfn;
        return (void *)nvJitLinkGetLinkedCubinSize;
    } else if (strcmp(symbol, "nvJitLinkGetLinkedCubin") == 0) {
        real_nvJitLinkGetLinkedCubin = (nvJitLinkGetLinkedCubin_fn)pfn;
        return (void *)nvJitLinkGetLinkedCubin;
    } else if (strcmp(symbol, "__nvJitLinkCreate_12_8") == 0) {
        real___nvJitLinkCreate_12_8 = (nvJitLinkCreate_fn)pfn;
        return (void *)__nvJitLinkCreate_12_8;
    } else if (strcmp(symbol, "__nvJitLinkAddData_12_8") == 0) {
        real___nvJitLinkAddData_12_8 = (nvJitLinkAddData_fn)pfn;
        return (void *)__nvJitLinkAddData_12_8;
    } else if (strcmp(symbol, "__nvJitLinkComplete_12_8") == 0) {
        real___nvJitLinkComplete_12_8 = (nvJitLinkComplete_fn)pfn;
        return (void *)__nvJitLinkComplete_12_8;
    } else if (strcmp(symbol, "__nvJitLinkGetLinkedCubinSize_12_8") == 0) {
        real___nvJitLinkGetLinkedCubinSize_12_8 = (nvJitLinkGetLinkedCubinSize_fn)pfn;
        return (void *)__nvJitLinkGetLinkedCubinSize_12_8;
    } else if (strcmp(symbol, "__nvJitLinkGetLinkedCubin_12_8") == 0) {
        real___nvJitLinkGetLinkedCubin_12_8 = (nvJitLinkGetLinkedCubin_fn)pfn;
        return (void *)__nvJitLinkGetLinkedCubin_12_8;
    }
    return pfn;
}

static void replace_driver_symbol(const char *symbol, void **pfn) {
    if (!symbol || !pfn || !*pfn) return;
    *pfn = wrapped_symbol(symbol, *pfn);
}

void *dlsym(void *handle, const char *symbol) {
    dlsym_fn real_dlsym = get_real_dlsym();
    if (!real_dlsym) {
        fprintf(stderr, "[dump_cuda_modules] failed to resolve real dlsym\n");
        return NULL;
    }
    void *pfn = real_dlsym(handle, symbol);
    void *wrapped = wrapped_symbol(symbol, pfn);
    if (symbol && (strncmp(symbol, "cu", 2) == 0 || strstr(symbol, "JitLink") != NULL)) {
        fprintf(stderr, "[dump_cuda_modules] dlsym symbol=%s pfn=%p%s\n", symbol, pfn, wrapped != pfn ? " wrapped" : "");
    }
    return wrapped;
}

CUresult cuModuleLoadData(CUmodule *module, const void *image) {
    if (!real_cuModuleLoadData) real_cuModuleLoadData = (cuModuleLoadData_fn)load_next("cuModuleLoadData");
    dump_inferred("cuModuleLoadData", "image", image);
    return real_cuModuleLoadData(module, image);
}

CUresult cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int numOptions, CUjit_option *options, void **optionValues) {
    if (!real_cuModuleLoadDataEx) real_cuModuleLoadDataEx = (cuModuleLoadDataEx_fn)load_next("cuModuleLoadDataEx");
    dump_inferred("cuModuleLoadDataEx", "image", image);
    return real_cuModuleLoadDataEx(module, image, numOptions, options, optionValues);
}

CUresult cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin) {
    if (!real_cuModuleLoadFatBinary) real_cuModuleLoadFatBinary = (cuModuleLoadFatBinary_fn)load_next("cuModuleLoadFatBinary");
    dump_inferred("cuModuleLoadFatBinary", "fatCubin", fatCubin);
    return real_cuModuleLoadFatBinary(module, fatCubin);
}

CUresult cuLibraryLoadData(CUlibrary *library, const void *code, CUjit_option *jitOptions, void **jitOptionsValues, unsigned int numJitOptions, CUlibraryOption *libraryOptions, void **libraryOptionValues, unsigned int numLibraryOptions) {
    if (!real_cuLibraryLoadData) real_cuLibraryLoadData = (cuLibraryLoadData_fn)load_next("cuLibraryLoadData");
    dump_inferred("cuLibraryLoadData", "code", code);
    return real_cuLibraryLoadData(library, code, jitOptions, jitOptionsValues, numJitOptions, libraryOptions, libraryOptionValues, numLibraryOptions);
}

CUresult cuLibraryLoadFromFile(CUlibrary *library, const char *fileName, CUjit_option *jitOptions, void **jitOptionsValues, unsigned int numJitOptions, CUlibraryOption *libraryOptions, void **libraryOptionValues, unsigned int numLibraryOptions) {
    if (!real_cuLibraryLoadFromFile) real_cuLibraryLoadFromFile = (cuLibraryLoadFromFile_fn)load_next("cuLibraryLoadFromFile");
    fprintf(stderr, "[dump_cuda_modules] cuLibraryLoadFromFile path=%s\n", fileName ? fileName : "(null)");
    return real_cuLibraryLoadFromFile(library, fileName, jitOptions, jitOptionsValues, numJitOptions, libraryOptions, libraryOptionValues, numLibraryOptions);
}

CUresult cuLinkAddData(CUlinkState state, CUjitInputType type, void *data, size_t size, const char *name, unsigned int numOptions, CUjit_option *options, void **optionValues) {
    if (!real_cuLinkAddData) real_cuLinkAddData = (cuLinkAddData_fn)load_next("cuLinkAddData");
    char label[256];
    snprintf(label, sizeof(label), "type%d_%s", type, name ? name : "anon");
    dump_blob("cuLinkAddData", label, data, size);
    return real_cuLinkAddData(state, type, data, size, name, numOptions, options, optionValues);
}

CUresult cuLinkAddData_v2(CUlinkState state, CUjitInputType type, void *data, size_t size, const char *name, unsigned int numOptions, CUjit_option *options, void **optionValues) {
    if (!real_cuLinkAddData_v2) real_cuLinkAddData_v2 = (cuLinkAddData_fn)load_next("cuLinkAddData_v2");
    char label[256];
    snprintf(label, sizeof(label), "type%d_%s", type, name ? name : "anon");
    dump_blob("cuLinkAddData_v2", label, data, size);
    return real_cuLinkAddData_v2(state, type, data, size, name, numOptions, options, optionValues);
}

CUresult cuLinkComplete(CUlinkState state, void **cubinOut, size_t *sizeOut) {
    if (!real_cuLinkComplete) real_cuLinkComplete = (cuLinkComplete_fn)load_next("cuLinkComplete");
    CUresult result = real_cuLinkComplete(state, cubinOut, sizeOut);
    if (result == 0 && cubinOut && *cubinOut && sizeOut) {
        dump_blob("cuLinkComplete", "linked", *cubinOut, *sizeOut);
    }
    return result;
}

CUresult cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags, CUdriverProcAddressQueryResult *symbolStatus) {
    if (!real_cuGetProcAddress_v2) real_cuGetProcAddress_v2 = (cuGetProcAddress_v2_fn)load_next("cuGetProcAddress_v2");
    CUresult result = real_cuGetProcAddress_v2(symbol, pfn, cudaVersion, flags, symbolStatus);
    if (result == 0) replace_driver_symbol(symbol, pfn);
    return result;
}

CUresult cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags) {
    if (!real_cuGetProcAddress_legacy) real_cuGetProcAddress_legacy = (cuGetProcAddress_legacy_fn)load_next("cuGetProcAddress");
    CUresult result = real_cuGetProcAddress_legacy(symbol, pfn, cudaVersion, flags);
    if (result == 0) replace_driver_symbol(symbol, pfn);
    return result;
}

nvJitLinkResult nvJitLinkCreate(nvJitLinkHandle *handle, uint32_t numOptions, const char **options) {
    if (!real_nvJitLinkCreate) real_nvJitLinkCreate = (nvJitLinkCreate_fn)load_next("nvJitLinkCreate");
    nvJitLinkResult result = real_nvJitLinkCreate(handle, numOptions, options);
    fprintf(stderr, "[dump_cuda_modules] nvJitLinkCreate handle=%p options=%u result=%d\n", handle ? (void *)*handle : NULL, numOptions, result);
    return result;
}

nvJitLinkResult nvJitLinkAddData(nvJitLinkHandle handle, nvJitLinkInputType inputType, const void *data, size_t size, const char *name) {
    if (!real_nvJitLinkAddData) real_nvJitLinkAddData = (nvJitLinkAddData_fn)load_next("nvJitLinkAddData");
    char label[256];
    snprintf(label, sizeof(label), "type%d_%s", inputType, name ? name : "anon");
    dump_blob("nvJitLinkAddData", label, data, size);
    return real_nvJitLinkAddData(handle, inputType, data, size, name);
}

nvJitLinkResult nvJitLinkComplete(nvJitLinkHandle handle) {
    if (!real_nvJitLinkComplete) real_nvJitLinkComplete = (nvJitLinkComplete_fn)load_next("nvJitLinkComplete");
    nvJitLinkResult result = real_nvJitLinkComplete(handle);
    fprintf(stderr, "[dump_cuda_modules] nvJitLinkComplete handle=%p result=%d\n", (void *)handle, result);
    return result;
}

nvJitLinkResult nvJitLinkGetLinkedCubinSize(nvJitLinkHandle handle, size_t *size) {
    if (!real_nvJitLinkGetLinkedCubinSize) real_nvJitLinkGetLinkedCubinSize = (nvJitLinkGetLinkedCubinSize_fn)load_next("nvJitLinkGetLinkedCubinSize");
    return real_nvJitLinkGetLinkedCubinSize(handle, size);
}

nvJitLinkResult nvJitLinkGetLinkedCubin(nvJitLinkHandle handle, void *cubin) {
    if (!real_nvJitLinkGetLinkedCubin) real_nvJitLinkGetLinkedCubin = (nvJitLinkGetLinkedCubin_fn)load_next("nvJitLinkGetLinkedCubin");
    if (!real_nvJitLinkGetLinkedCubinSize) real_nvJitLinkGetLinkedCubinSize = (nvJitLinkGetLinkedCubinSize_fn)load_next("nvJitLinkGetLinkedCubinSize");
    nvJitLinkResult result = real_nvJitLinkGetLinkedCubin(handle, cubin);
    size_t size = 0;
    if (result == 0 && real_nvJitLinkGetLinkedCubinSize && real_nvJitLinkGetLinkedCubinSize(handle, &size) == 0) {
        dump_blob("nvJitLinkGetLinkedCubin", "linked", cubin, size);
    }
    return result;
}

nvJitLinkResult __nvJitLinkCreate_12_8(nvJitLinkHandle *handle, uint32_t numOptions, const char **options) {
    if (!real___nvJitLinkCreate_12_8) real___nvJitLinkCreate_12_8 = (nvJitLinkCreate_fn)load_next("__nvJitLinkCreate_12_8");
    nvJitLinkResult result = real___nvJitLinkCreate_12_8(handle, numOptions, options);
    fprintf(stderr, "[dump_cuda_modules] __nvJitLinkCreate_12_8 handle=%p options=%u result=%d\n", handle ? (void *)*handle : NULL, numOptions, result);
    return result;
}

nvJitLinkResult __nvJitLinkAddData_12_8(nvJitLinkHandle handle, nvJitLinkInputType inputType, const void *data, size_t size, const char *name) {
    if (!real___nvJitLinkAddData_12_8) real___nvJitLinkAddData_12_8 = (nvJitLinkAddData_fn)load_next("__nvJitLinkAddData_12_8");
    char label[256];
    snprintf(label, sizeof(label), "type%d_%s", inputType, name ? name : "anon");
    dump_blob("__nvJitLinkAddData_12_8", label, data, size);
    return real___nvJitLinkAddData_12_8(handle, inputType, data, size, name);
}

nvJitLinkResult __nvJitLinkComplete_12_8(nvJitLinkHandle handle) {
    if (!real___nvJitLinkComplete_12_8) real___nvJitLinkComplete_12_8 = (nvJitLinkComplete_fn)load_next("__nvJitLinkComplete_12_8");
    nvJitLinkResult result = real___nvJitLinkComplete_12_8(handle);
    fprintf(stderr, "[dump_cuda_modules] __nvJitLinkComplete_12_8 handle=%p result=%d\n", (void *)handle, result);
    return result;
}

nvJitLinkResult __nvJitLinkGetLinkedCubinSize_12_8(nvJitLinkHandle handle, size_t *size) {
    if (!real___nvJitLinkGetLinkedCubinSize_12_8) real___nvJitLinkGetLinkedCubinSize_12_8 = (nvJitLinkGetLinkedCubinSize_fn)load_next("__nvJitLinkGetLinkedCubinSize_12_8");
    return real___nvJitLinkGetLinkedCubinSize_12_8(handle, size);
}

nvJitLinkResult __nvJitLinkGetLinkedCubin_12_8(nvJitLinkHandle handle, void *cubin) {
    if (!real___nvJitLinkGetLinkedCubin_12_8) real___nvJitLinkGetLinkedCubin_12_8 = (nvJitLinkGetLinkedCubin_fn)load_next("__nvJitLinkGetLinkedCubin_12_8");
    if (!real___nvJitLinkGetLinkedCubinSize_12_8) real___nvJitLinkGetLinkedCubinSize_12_8 = (nvJitLinkGetLinkedCubinSize_fn)load_next("__nvJitLinkGetLinkedCubinSize_12_8");
    nvJitLinkResult result = real___nvJitLinkGetLinkedCubin_12_8(handle, cubin);
    size_t size = 0;
    if (result == 0 && real___nvJitLinkGetLinkedCubinSize_12_8 && real___nvJitLinkGetLinkedCubinSize_12_8(handle, &size) == 0) {
        dump_blob("__nvJitLinkGetLinkedCubin_12_8", "linked", cubin, size);
    }
    return result;
}
