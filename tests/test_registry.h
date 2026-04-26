#ifndef AB_TEST_REGISTRY_H
#define AB_TEST_REGISTRY_H

typedef struct {
    const char *name;
    void (*fn)(void);
} TestCase;

#define TEST_REGISTRY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))

#endif /* AB_TEST_REGISTRY_H */
