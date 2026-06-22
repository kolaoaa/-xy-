#ifndef __PLANAR_ACTUATOR_H
#define __PLANAR_ACTUATOR_H

#include "main.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef ACTUATOR_GPIO_PORT
#define ACTUATOR_GPIO_PORT GPIOE
#endif

#ifndef ACTUATOR_GPIO_PIN
#define ACTUATOR_GPIO_PIN GPIO_PIN_5
#endif

#ifndef ACTUATOR_ACTIVE_LEVEL
#define ACTUATOR_ACTIVE_LEVEL GPIO_PIN_SET
#endif

void actuator_init(void);
void actuator_on(void);
void actuator_off(void);
uint8_t actuator_is_on(void);

#ifdef __cplusplus
}
#endif

#endif
