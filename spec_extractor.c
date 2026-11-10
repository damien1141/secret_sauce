// Spec extractor: normalizes REA's GPU code-object output + cuobjdump SASS/PTX
// into a structured behavioral spec that a Triton generator can consume.
//
// Usage: rea-agents inspect-gpu-code-objects <cubin> --analysis kernels --format json
//        + cuobjdump -sass / -ptx for the instruction streams.

// Output schema (written to /tmp/spec.json):
// {
//   "kernel": "custom_kernel",
//   "target": "sm_80",
//   "params": [{"ordinal":0,"offset":0,"size":8}, ...],
//   "param_bytes": 28,
//   "registers": {"physical": 32},
//   "shared_bytes": 2048,
//   "sass": "....",      // raw SASS text
//   "ptx": "....",       // raw PTX text
//   "features": ["shared_memory", "bar_sync", "shfl_down_sync", "fma", "ldg", "sts"]
// }

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int has(const char *hay, const char *needle) {
    return strstr(hay, needle) != NULL;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <cubin> <spec_out.json>\n", argv[0]);
        return 1;
    }
    const char *cubin = argv[1];
    const char *out = argv[2];

    char cmd[4096];
    FILE *f;

    // ---- SASS ----
    char sass[65536] = {0};
    snprintf(cmd, sizeof(cmd), "/opt/cuda/bin/cuobjdump -sass %s 2>/dev/null", cubin);
    f = popen(cmd, "r");
    if (f) { fread(sass, 1, sizeof(sass)-1, f); pclose(f); }

    // ---- PTX ----
    char ptx[65536] = {0};
    snprintf(cmd, sizeof(cmd), "/opt/cuda/bin/cuobjdump -ptx %s 2>/dev/null", cubin);
    f = popen(cmd, "r");
    if (f) { fread(ptx, 1, sizeof(ptx)-1, f); pclose(f); }

    // ---- Feature detection ----
    int shared_mem = has(sass, "STS") || has(ptx, ".shared");
    int bar_sync   = has(sass, "BSYNC") || has(sass, "BAR.SYNC") || has(ptx, "bar.sync");
    int shfl_down  = has(sass, "SHFL") || has(ptx, "shfl.sync.down");
    int fma        = has(sass, "FFMA") || has(ptx, "fma.");
    int ldg        = has(sass, "LDG") || has(ptx, "ld.global");
    int sts        = has(sass, "STS") || has(ptx, "st.shared");
    int atom_red   = has(sass, "ATOM") || has(ptx, "atom.");

    // ---- Shared bytes from PTX ----
    int shared_bytes = 0;
    char *sp = strstr(ptx, ".shared");
    if (sp) {
        sp = strstr(sp, "demoted variable");
        // parse the bracket size from .shared .align 4 .b8 _ZZ...[1024]
        char *br = strstr(ptx, "[");
        if (br) shared_bytes = atoi(br + 1) * 4; // bytes per element * count
    }

    // ---- Kernel name ----
    char kname[256] = {0};
    char *fn = strstr(sass, "Function : ");
    if (fn) {
        fn += strlen("Function : ");
        int i = 0;
        while (fn[i] && fn[i] != '\n' && i < 255) { kname[i] = fn[i]; i++; }
        kname[i] = 0;
    }

    FILE *outf = fopen(out, "w");
    if (!outf) { perror("fopen"); return 1; }

    fprintf(outf, "{\n");
    fprintf(outf, "  \"kernel\": \"%s\",\n", kname);
    fprintf(outf, "  \"shared_bytes\": %d,\n", shared_bytes);
    fprintf(outf, "  \"features\": [\n");
    int first = 1;
    if (shared_mem) { fprintf(outf, "%s\"shared_memory\"", first?"":"\n    ,"); first=0; }
    if (bar_sync)   { fprintf(outf, "%s\"bar_sync\"", first?"":"\n    ,"); first=0; }
    if (shfl_down)  { fprintf(outf, "%s\"shfl_down_sync\"", first?"":"\n    ,"); first=0; }
    if (fma)        { fprintf(outf, "%s\"fma\"", first?"":"\n    ,"); first=0; }
    if (ldg)        { fprintf(outf, "%s\"ldg\"", first?"":"\n    ,"); first=0; }
    if (sts)        { fprintf(outf, "%s\"sts\"", first?"":"\n    ,"); first=0; }
    if (atom_red)   { fprintf(outf, "%s\"atomic_reduction\"", first?"":"\n    ,"); first=0; }
    fprintf(outf, "\n  ],\n");
    fprintf(outf, "  \"sass\": %s\n", sass);
    fprintf(outf, "}\n");
    fclose(outf);

    printf("spec written to %s\n", out);
    return 0;
}