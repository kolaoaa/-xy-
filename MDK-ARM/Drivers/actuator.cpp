#include "actuator.h"

static uint8_t actuator_state = 0U;

static GPIO_PinState inactive_level(void)
{
    return (ACTUATOR_ACTIVE_LEVEL == GPIO_PIN_SET) ? GPIO_PIN_RESET : GPIO_PIN_SET;
}

void actuator_init(void)
{
    GPIO_InitTypeDef GPIO_InitStruct = {0};

    __HAL_RCC_GPIOE_CLK_ENABLE();
    HAL_GPIO_WritePin(ACTUATOR_GPIO_PORT, ACTUATOR_GPIO_PIN, inactive_level());

    GPIO_InitStruct.Pin = ACTUATOR_GPIO_PIN;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(ACTUATOR_GPIO_PORT, &GPIO_InitStruct);

    actuator_state = 0U;
}

void actuator_on(void)
{
    HAL_GPIO_WritePin(ACTUATOR_GPIO_PORT, ACTUATOR_GPIO_PIN, ACTUATOR_ACTIVE_LEVEL);
    actuator_state = 1U;
}

void actuator_off(void)
{
    HAL_GPIO_WritePin(ACTUATOR_GPIO_PORT, ACTUATOR_GPIO_PIN, inactive_level());
    actuator_state = 0U;
}

uint8_t actuator_is_on(void)
{
    return actuator_state;
}
