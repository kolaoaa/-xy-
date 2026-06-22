/**
  ******************************************************************************
  * @file           :
  * @author         : Xiang Guo
  * @brief          : brief
  * @date           : 2023/05/07
  ******************************************************************************
  * @attention
  *
	*
  ******************************************************************************
  */
#ifndef __MY_CONFIG_H
#define __MY_CONFIG_H

/* ------------------------------ Includes ------------------------------ */

#include "xkey.h"
#include "xLinearModule.h"
#include "XYplatform.h"
/* ------------------------------ Defines ------------------------------ */

/* 夹子舵机配置：J9 SERVO1 -> PA0 */
#define SERVO_PWM_GPIO_PORT GPIOA
#define SERVO_PWM_GPIO_PIN GPIO_PIN_0
#define SERVO_MIN_POS 500
#define SERVO_MID_POS 1500
#define SERVO_MAX_POS 1700
#define SERVO_DEVIATION_US 0
#define SERVO_DEVIATION_LIMIT 100
#define SERVO_FRAME_US 20000
#define SERVO_STEP_MIN_US 20
#define SERVO_OPEN_POS 500
#define SERVO_CLOSE_POS 1700
#define SERVO_ACTION_TIME 1000

/* ------------------------------ Variable Declarations ------------------------------ */

extern xkey::Key g_key[4];
extern x_linear_module::LinearModule g_linearModule[2];
extern xy_platform::XYplatform g_xyPlatform;


/* ------------------------------ Typedef ------------------------------ */

/* ------------------------------ Class ------------------------------ */


#endif
