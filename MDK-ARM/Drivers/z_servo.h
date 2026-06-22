#ifndef Z_SERVO_H
#define Z_SERVO_H

#include "main.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void servo_init(void);
void servo_set_deviation(int16_t deviation_us);
void servo_middle(void);
void servo_open(void);
void servo_close(void);
void servo_update_frame(void);

#ifdef __cplusplus
}
#endif

#endif
