#include "z_servo.h"
#include "cmsis_os2.h"
#include "my_config.h"

#ifndef SERVO_PWM_GPIO_PORT
#define SERVO_PWM_GPIO_PORT GPIOA
#endif

#ifndef SERVO_PWM_GPIO_PIN
#define SERVO_PWM_GPIO_PIN GPIO_PIN_0
#endif

#ifndef SERVO_MIN_POS
#define SERVO_MIN_POS 500
#endif

#ifndef SERVO_MID_POS
#define SERVO_MID_POS 1500
#endif

#ifndef SERVO_MAX_POS
#define SERVO_MAX_POS 1700
#endif

#ifndef SERVO_DEVIATION_US
#define SERVO_DEVIATION_US 0
#endif

#ifndef SERVO_DEVIATION_LIMIT
#define SERVO_DEVIATION_LIMIT 100
#endif

#ifndef SERVO_FRAME_US
#define SERVO_FRAME_US 20000
#endif

#ifndef SERVO_STEP_MIN_US
#define SERVO_STEP_MIN_US 20
#endif

static volatile uint16_t servo_target_us = SERVO_MID_POS;
static volatile int16_t servo_deviation_us = SERVO_DEVIATION_US;
static uint16_t servo_current_us = SERVO_MID_POS;

static void servo_enable_gpio_clock(GPIO_TypeDef *port)
{
    if (port == GPIOA)
    {
        __HAL_RCC_GPIOA_CLK_ENABLE();
    }
    else if (port == GPIOB)
    {
        __HAL_RCC_GPIOB_CLK_ENABLE();
    }
    else if (port == GPIOC)
    {
        __HAL_RCC_GPIOC_CLK_ENABLE();
    }
    else if (port == GPIOD)
    {
        __HAL_RCC_GPIOD_CLK_ENABLE();
    }
    else if (port == GPIOE)
    {
        __HAL_RCC_GPIOE_CLK_ENABLE();
    }
}

static void servo_dwt_init(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0U;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

static void servo_delay_us(uint32_t us)
{
    const uint32_t cycles_per_us = SystemCoreClock / 1000000U;
    const uint32_t delay_cycles = cycles_per_us * us;
    const uint32_t start = DWT->CYCCNT;

    while ((DWT->CYCCNT - start) < delay_cycles)
    {
    }
}

static uint16_t servo_clamp_pulse(int32_t pulse_us)
{
    if (pulse_us < SERVO_MIN_POS)
    {
        return (uint16_t)SERVO_MIN_POS;
    }

    if (pulse_us > SERVO_MAX_POS)
    {
        return (uint16_t)SERVO_MAX_POS;
    }

    return (uint16_t)pulse_us;
}

static uint16_t servo_with_deviation(uint16_t base_us)
{
    return servo_clamp_pulse((int32_t)base_us + servo_deviation_us);
}

static void servo_set_target(uint16_t pulse_us)
{
    servo_target_us = servo_with_deviation(pulse_us);
}

static uint16_t servo_next_pulse(uint16_t current_us, uint16_t target_us)
{
    const uint16_t diff = (current_us > target_us)
        ? (uint16_t)(current_us - target_us)
        : (uint16_t)(target_us - current_us);
    uint16_t step_us = SERVO_STEP_MIN_US;

    if (SERVO_ACTION_TIME >= 20)
    {
        const uint16_t frames = (uint16_t)(SERVO_ACTION_TIME / 20U);
        const uint16_t calculated_step = (frames == 0U) ? diff : (uint16_t)(diff / frames);
        if (calculated_step > step_us)
        {
            step_us = calculated_step;
        }
    }

    if (diff <= step_us)
    {
        return target_us;
    }

    return (current_us < target_us)
        ? (uint16_t)(current_us + step_us)
        : (uint16_t)(current_us - step_us);
}

void servo_init(void)
{
    GPIO_InitTypeDef GPIO_InitStruct = {0};

    servo_enable_gpio_clock(SERVO_PWM_GPIO_PORT);

    GPIO_InitStruct.Pin = SERVO_PWM_GPIO_PIN;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(SERVO_PWM_GPIO_PORT, &GPIO_InitStruct);
    HAL_GPIO_WritePin(SERVO_PWM_GPIO_PORT, SERVO_PWM_GPIO_PIN, GPIO_PIN_RESET);

    servo_dwt_init();

    servo_deviation_us = SERVO_DEVIATION_US;
    servo_current_us = servo_with_deviation(SERVO_MID_POS);
    servo_target_us = servo_current_us;
}

void servo_set_deviation(int16_t deviation_us)
{
    if (deviation_us > SERVO_DEVIATION_LIMIT)
    {
        deviation_us = SERVO_DEVIATION_LIMIT;
    }
    else if (deviation_us < -SERVO_DEVIATION_LIMIT)
    {
        deviation_us = -SERVO_DEVIATION_LIMIT;
    }

    servo_deviation_us = deviation_us;
}

void servo_middle(void)
{
    servo_set_target(SERVO_MID_POS);
}

void servo_open(void)
{
    servo_set_target(SERVO_OPEN_POS);
}

void servo_close(void)
{
    servo_set_target(SERVO_CLOSE_POS);
}

void servo_update_frame(void)
{
    uint32_t rest_us;

    servo_current_us = servo_next_pulse(servo_current_us, servo_target_us);

    HAL_GPIO_WritePin(SERVO_PWM_GPIO_PORT, SERVO_PWM_GPIO_PIN, GPIO_PIN_SET);
    servo_delay_us(servo_current_us);
    HAL_GPIO_WritePin(SERVO_PWM_GPIO_PORT, SERVO_PWM_GPIO_PIN, GPIO_PIN_RESET);

    rest_us = SERVO_FRAME_US - servo_current_us;
    if (rest_us >= 1000U)
    {
        osDelay(rest_us / 1000U);
        rest_us %= 1000U;
    }

    if (rest_us > 0U)
    {
        servo_delay_us(rest_us);
    }
}
