/**
 * Copyright 2022 by Au-Zone Technologies.  All Rights Reserved.
 *
 * Software that is described herein is for illustrative purposes only which
 * provides customers with programming information regarding the DeepView VAAL
 * library. This software is supplied "AS IS" without any warranties of any
 * kind, and Au-Zone Technologies and its licensor disclaim any and all
 * warranties, express or implied, including all implied warranties of
 * merchantability, fitness for a particular purpose and non-infringement of
 * intellectual property rights.  Au-Zone Technologies assumes no responsibility
 * or liability for the use of the software, conveys no license or rights under
 * any patent, copyright, mask work right, or any other intellectual property
 * rights in or to any products. Au-Zone Technologies reserves the right to make
 * changes in the software without notification. Au-Zone Technologies also makes
 * no representation or warranty that such application will be suitable for the
 * specified use without further testing or modification.
 */

#include <errno.h>
#include <getopt.h>
#include <stdio.h>
#include <string.h>

#ifndef _WIN32
#include <strings.h>
#endif

#include "vaal.h"

#define USAGE \
    "detect [hv] model.rtm image0 [imageN]\n\
    -h, --help\n\
        Display help information\n\
    -v, --version\n\
        Display version information\n\
    -e, --engine\n\
        Compute engine type \"cpu\", \"npu\"\n\
    -n, --norm\n\
        Normalization method applied to input images. \n\
            - raw (default, no processing) \n\
            - unsigned (0...1) \n\
            - signed (-1...1) \n\
            - whitening (per-image standardization/whitening) \n\
            - imagenet (standardization using imagenet) \n\
"

int
main(int argc, char* argv[])
{
    // These can be modified as needed
    int         err;
    const char* engine        = "npu";
    const char* model         = NULL;
    int         norm          = 0;
    int         max_label     = 16;
    int         max_keypoints = 50;

    static struct option options[] = {
        {"help", no_argument, NULL, 'h'},
        {"version", no_argument, NULL, 'v'},
        {"norm", required_argument, NULL, 'n'},
        {"engine", required_argument, NULL, 'e'},
    };

    // Processing of command line arguments
    for (;;) {
        int opt =
            getopt_long(argc, argv, "hvn:e:", options, NULL);
        if (opt == -1) break;

        switch (opt) {
        case 'h':
            printf(USAGE);
            return EXIT_SUCCESS;
        case 'v':
            printf("DeepView VisionPack Detection Sample with VAAL %s\n",
                   vaal_version(NULL, NULL, NULL, NULL));
            return EXIT_SUCCESS;
        case 'n':
            if (strcmp(optarg, "raw") == 0) {
                norm = 0;
            } else if (strcmp(optarg, "signed") == 0) {
                norm = VAAL_IMAGE_PROC_SIGNED_NORM;
            } else if (strcmp(optarg, "unsigned") == 0) {
                norm = VAAL_IMAGE_PROC_UNSIGNED_NORM;
            } else if (strcmp(optarg, "whitening") == 0) {
                norm = VAAL_IMAGE_PROC_WHITENING;
            } else if (strcmp(optarg, "imagenet") == 0) {
                norm = VAAL_IMAGE_PROC_IMAGENET;
            } else {
                fprintf(stderr,
                        "unsupported image normalization method: %s\n",
                        optarg);
                return EXIT_FAILURE;
            }
            break;
        case 'e':
            engine = optarg;
            break;
        default:
            fprintf(stderr,
                    "invalid parameter %c, try --help for usage\n",
                    opt);
            return EXIT_FAILURE;
        }
    }

    if (argv[optind] == NULL) {
        fprintf(stderr, "missing required model, try --help for usage\n");
        return EXIT_FAILURE;
    }

    model = argv[optind++];

    // Initialize boxes object and context with requested engine
    size_t        num_kpts  = 0;
    VAALKeypoint* keypoints = calloc(max_keypoints, sizeof(VAALKeypoint));
    VAALContext  *pose_ctx  = NULL;

    pose_ctx = vaal_context_create(engine);
    err = vaal_load_model_file(pose_ctx, model);
    if (err) {
        vaal_context_release(pose_ctx);
        pose_ctx = vaal_model_probe(engine, model);
        if (!pose_ctx) {
            fprintf(stderr, "failed to load model: %s\n", vaal_strerror(err));
        return EXIT_FAILURE;
        }
    }

    vaal_parameter_seti(pose_ctx, "normalization", &norm, 1);

    // Loop through all provided images
    for (int i = optind; i < argc; i++) {
        int64_t     start, load_ns, inference_ns, boxes_ns, pose_ns;
        const char* image = argv[i];

        // Load image into context
        start = vaal_clock_now();
        err   = vaal_load_image_file(pose_ctx, NULL, image, NULL, 0);
        if (err) {
            fprintf(stderr,
                    "failed to load %s: %s\n",
                    image,
                    vaal_strerror(err));
            return EXIT_FAILURE;
        }
        load_ns = vaal_clock_now() - start;

        start        = vaal_clock_now();
        err          = vaal_run_model(pose_ctx);
        inference_ns = vaal_clock_now() - start;
        if (err) {
            fprintf(stderr, "failed to run model: %s\n", vaal_strerror(err));
            return EXIT_FAILURE;
        }

        start = vaal_clock_now();
        if (vaal_keypoints(pose_ctx, keypoints, max_keypoints, &num_kpts)) {
            printf("Keypoint detection failed.\n");
            return EXIT_FAILURE;
        }
        pose_ns = vaal_clock_now() - start;

        printf("Load: %.4f Infer: %.4f Decode: %.4f\n",
               load_ns / 1e6,
               inference_ns / 1e6,
               pose_ns / 1e6);
        for (size_t j = 0; j < num_kpts; j++) {
            const VAALKeypoint* point = &keypoints[j];

            printf("  [%3zu] - (%3d%%): %3.2f %3.2f\r\n",
                j,
                (int) (point->score * 100),
                point->x,
                point->y);
        }
    }

    // Free memory used for context and boxes
    vaal_context_release(pose_ctx);
    free(keypoints);

    return EXIT_SUCCESS;
}
