/**
 ******************************************************************************
 * @file           :
 * @author         : Xiang Guo
 * @date           : 2023/xx/xx
 * @brief          :
 ******************************************************************************
 * @attention
 *
 *
 ******************************************************************************
 */

/* ------------------------------ Includes ------------------------------ */

#include "XYplatform.h"
#include "xLinearModule.h"

/* ------------------------------ Defines ------------------------------ */

#define abs(x) ((x) > 0 ? (x) : -(x))
#define POSITION_ERROR_THRESHOLD 0.01f
#define DEG_TO_RAD 0.01745329251994329577f
#define FAST_TRIG_PI 3.14159265358979323846f
#define FAST_TRIG_HALF_PI 1.57079632679489661923f
#define FAST_TRIG_TWO_PI 6.28318530717958647692f

/* ------------------------------ variables ------------------------------ */

/* ------------------------------ Functions ------------------------------ */

static float LinearInterJudge(float x_real, float y_real, float x_target,
                              float y_target, float x_start, float y_start) {
  float x_e = x_target - x_start;
  float y_e = y_target - y_start;
  float x_i = x_real - x_start;
  float y_i = y_real - y_start;
  float f = x_e * y_i - x_i * y_e;
  return f;
}
static float CircularInterJudge(float x_real, float y_real, float center_x, float center_y, float radius) {
  float x_c = x_real - center_x;
  float y_c = y_real - center_y;
  float f = x_c * x_c + y_c * y_c - radius * radius;
  return f;
}

static float FastAbsF(float value) {
  return value >= 0.0f ? value : -value;
}

static float WrapRadians(float radians) {
  while (radians > FAST_TRIG_PI) {
    radians -= FAST_TRIG_TWO_PI;
  }
  while (radians < -FAST_TRIG_PI) {
    radians += FAST_TRIG_TWO_PI;
  }
  return radians;
}

static float FastSinF(float radians) {
  radians = WrapRadians(radians);
  float y = (4.0f / FAST_TRIG_PI) * radians -
            (4.0f / (FAST_TRIG_PI * FAST_TRIG_PI)) * radians * FastAbsF(radians);
  return y + 0.225f * (y * FastAbsF(y) - y);
}

static float FastCosF(float radians) {
  return FastSinF(radians + FAST_TRIG_HALF_PI);
}

namespace xy_platform {
XYplatform::XYplatform(x_linear_module::LinearModule *x,
                       x_linear_module::LinearModule *y, float inter_step, float pid_limit_output,
                      float pid_kp, float pid_ki, float pid_kd,
                      float pid_time_period_s)
    : pos_pid_x(pid_kp, pid_ki, pid_kd, pid_limit_output, pid_time_period_s),
      pos_pid_y(pid_kp, pid_ki, pid_kd, pid_limit_output, pid_time_period_s),
      inter_step(inter_step) {
  this->x = x;
  this->y = y;
}

void XYplatform::MotionConfig(int8_t x_dir, int8_t y_dir, float max_vel,float acc) {
  this->x->MotionConfig(x_dir, max_vel, acc);
  this->y->MotionConfig(y_dir, max_vel, acc);
  this->max_vel = max_vel;
  this->acc = acc;
  this->x_dir = x_dir;
  this->y_dir = y_dir;
}

void XYplatform::ConfigureWorkspace(float x_min, float x_max, float y_min, float y_max) {
  this->x_min = x_min;
  this->x_max = x_max;
  this->y_min = y_min;
  this->y_max = y_max;
}

bool XYplatform::IsWithinWorkspace(float x, float y) const {
  return x >= this->x_min && x <= this->x_max &&
         y >= this->y_min && y <= this->y_max;
}

bool XYplatform::IsHomed(void) const {
  return this->homed;
}

bool XYplatform::IsHoming(void) const {
  return this->mode == PLATFORM_MODE_FIND_HOME;
}

void XYplatform::ReportSafetyError(uint8_t error_code) {
  this->Stop();
  this->homed = false;
  this->error_code = error_code;
  this->x->SetMode(x_linear_module::MODULE_MODE_ERROR);
  this->y->SetMode(x_linear_module::MODULE_MODE_ERROR);
}

uint8_t XYplatform::GetErrorCode(void) const {
  return this->error_code;
}

void XYplatform::SetCurrentAsZero(void) {
  this->Stop();
  this->error_code = PLATFORM_ERROR_NONE;
  this->homed = true;
  this->x->SetPosition(0.0f);
  this->y->SetPosition(0.0f);
  this->x->SetTargetPosition(0.0f);
  this->y->SetTargetPosition(0.0f);
  this->x_real = 0.0f;
  this->y_real = 0.0f;
  this->x_target = 0.0f;
  this->y_target = 0.0f;
  this->x_interpolation_start = 0.0f;
  this->y_interpolation_start = 0.0f;
  this->x_interpolation_target = 0.0f;
  this->y_interpolation_target = 0.0f;
  this->x_interpolation_final = 0.0f;
  this->y_interpolation_final = 0.0f;
  this->linear_waiting_start = false;
  this->circular_waiting_start = false;
}

void XYplatform::FindHome(void) {
  this->error_code = PLATFORM_ERROR_NONE;
  this->homed = false;
  this->mode = PLATFORM_MODE_FIND_HOME;
  if (HAL_GPIO_ReadPin(this->x->limit_switch1_port, this->x->limit_switch1_pin) == GPIO_PIN_SET) {
    this->x->SetMode(x_linear_module::MODULE_MODE_POSITION);
    this->x->SetPosition(-10.0f);
    this->x->SetTargetPosition(0.0f);
  } else {
    this->x->SetMode(x_linear_module::MODULE_MODE_VELOCITY);
    this->x->SetTargetVelocity(-10.0f);
  }
  if (HAL_GPIO_ReadPin(this->y->limit_switch1_port, this->y->limit_switch1_pin) == GPIO_PIN_SET) {
    this->y->SetMode(x_linear_module::MODULE_MODE_POSITION);
    this->y->SetPosition(-10.0f);
    this->y->SetTargetPosition(0.0f);
  } else {
    this->y->SetMode(x_linear_module::MODULE_MODE_VELOCITY);
    this->y->SetTargetVelocity(-10.0f);
  }
}

void XYplatform::MoveTo(float x, float y) {
  this->MoveTo(x, y, this->max_vel);
}

void XYplatform::MoveTo(float x, float y, float vel) {
  if (!this->homed) {
    this->ReportSafetyError(PLATFORM_ERROR_NOT_HOMED);
    return;
  }
  if (!this->IsWithinWorkspace(x, y)) {
    this->ReportSafetyError(PLATFORM_ERROR_OUT_OF_RANGE);
    return;
  }
  // 设置模式为手动模式
  this->mode = PLATFORM_MODE_MANUAL;
  this->x_target = x;
  this->y_target = y;

  // 在位置模式下，底层会根据目标位置自动决定方向，这里只传速度幅值。
  float dx = x - this->x_real;
  float dy = y - this->y_real;
  float abs_dx = abs(dx);
  float abs_dy = abs(dy);
  float max_delta = abs_dx > abs_dy ? abs_dx : abs_dy;

  float speed = abs(vel);


  float vel_x = 0.0f;
  float vel_y = 0.0f;
  if (max_delta > 0.0f) {
    vel_x = speed * abs_dx / max_delta;
    vel_y = speed * abs_dy / max_delta;
  }

  // 设置x和y目标位置
  this->x->SetMode(x_linear_module::MODULE_MODE_POSITION);
  this->y->SetMode(x_linear_module::MODULE_MODE_POSITION);
  this->x->SetTargetPositionWithVelocity(x, vel_x);
  this->y->SetTargetPositionWithVelocity(y, vel_y);
}

void XYplatform::MoveRelative(float dx, float dy, float vel) {
  float current_x = this->x->GetPosition();
  float current_y = this->y->GetPosition();
  this->MoveTo(current_x + dx, current_y + dy, vel);
}

void XYplatform::LinearInterpolation(float x, float y, float vel, float step) {
  // 从当前位置开始线性插补
  this->LinearInterpolation(this->x_real, this->y_real, x, y, vel, step);
}

void XYplatform::LinearInterpolation(float x_start, float y_start, float x_end,float y_end, float vel, float step) {
  if (!this->homed) {
    this->ReportSafetyError(PLATFORM_ERROR_NOT_HOMED);
    return;
  }
  if (!this->IsWithinWorkspace(x_start, y_start) || !this->IsWithinWorkspace(x_end, y_end)) {
    this->ReportSafetyError(PLATFORM_ERROR_OUT_OF_RANGE);
    return;
  }
  // 先moveto到起始点
  this->MoveTo(x_start, y_start, vel);
  
  // 设置最终目标位置
  this->x_interpolation_final = x_end ;
  this->y_interpolation_final = y_end;
  
  // 设置插补参数
  this->inter_vel =vel;
  this->inter_step = abs(step);
  
  // 设置插补起始位置为起点
  this->x_interpolation_start = x_start;
  this->y_interpolation_start = y_start;
  this->x_interpolation_target = x_start;
  this->y_interpolation_target = y_start;
  
  // 标记正在等待到达起始点
  this->linear_waiting_start = true;
}

void XYplatform::CircularInterpolation(float center_x, float center_y,float radius, float vel, float angle_start,float angle_end, bool clockwise, float step) {
  if (!this->homed) {
    this->ReportSafetyError(PLATFORM_ERROR_NOT_HOMED);
    return;
  }
  if (radius < 0.0f ||
      !this->IsWithinWorkspace(center_x - radius, center_y - radius) ||
      !this->IsWithinWorkspace(center_x + radius, center_y + radius)) {
    this->ReportSafetyError(PLATFORM_ERROR_OUT_OF_RANGE);
    return;
  }
  float x_start = center_x + radius * FastCosF(angle_start*DEG_TO_RAD);
  float y_start = center_y + radius * FastSinF(angle_start*DEG_TO_RAD);
  this->MoveTo(x_start, y_start, vel);

  // 记录插补起始位置
  this->x_interpolation_start = x_start;
  this->y_interpolation_start = y_start;
  this->x_interpolation_target = x_start;
  this->y_interpolation_target = y_start;
  this->x_center = center_x;
  this->y_center = center_y;
  // 设置插补目标位置
  this->x_interpolation_final = center_x + radius * FastCosF(angle_end*DEG_TO_RAD);
  this->y_interpolation_final = center_y + radius * FastSinF(angle_end*DEG_TO_RAD);

  // 设置插补速度
  this->inter_vel = vel;
  // 设置插补步长
  this->inter_step = step;
  // 设置圆弧插补方向
  this->clockwise = clockwise;
  // 设置圆弧插补半径
  this->radius = radius;
  // 设置插补步长
  this->inter_step = abs(step);
  // 标记正在等待到达圆弧插补起始点
  this->circular_waiting_start = true;
}

void XYplatform::ClosedLoopControl(float x_pos_ref, float y_pos_ref) {
  if (!this->homed) {
    this->ReportSafetyError(PLATFORM_ERROR_NOT_HOMED);
    return;
  }
  if (!this->IsWithinWorkspace(x_pos_ref, y_pos_ref)) {
    this->ReportSafetyError(PLATFORM_ERROR_OUT_OF_RANGE);
    return;
  }
  // 设置模式为闭环位置控制模式
  this->mode = PLATFORM_MODE_CLOSED_LOOP;
  // 设置x和y目标位置
  this->x_target = x_pos_ref;
  this->y_target = y_pos_ref;

  this->x->SetMode(x_linear_module::MODULE_MODE_VELOCITY);
  this->y->SetMode(x_linear_module::MODULE_MODE_VELOCITY);
}

void XYplatform::ControlLoop(void) {
  // 请完成此函数 Start

  // 更新x和y的实际位置
  this->x_real = this->x->GetPosition();
  this->y_real = this->y->GetPosition();

  // 根据模式进行不同的控制
  if (this->mode == PLATFORM_MODE_IDLE) {
  } 
  else if (this->mode == PLATFORM_MODE_MANUAL) {
    if (abs(this->x_real - this->x_target) <= POSITION_ERROR_THRESHOLD &&
        abs(this->y_real - this->y_target) <= POSITION_ERROR_THRESHOLD) {
      this->mode =PLATFORM_MODE_IDLE;
    }
    // 检查是否在等待线性插补启动
    if (this->linear_waiting_start) {
      // 检查是否已到达起始点
      if (abs(this->x_real - this->x_interpolation_start) <= POSITION_ERROR_THRESHOLD &&
          abs(this->y_real - this->y_interpolation_start) <= POSITION_ERROR_THRESHOLD) {
        // 已到达起始点，切换到线性插补模式
        this->mode = PLATFORM_MODE_LINEAR_INTERPOLATION;
        this->x->SetMode(x_linear_module::MODULE_MODE_POSITION);
        this->y->SetMode(x_linear_module::MODULE_MODE_POSITION);
        this->x_target = this->x_interpolation_final;
        this->y_target = this->y_interpolation_final;
        this->linear_waiting_start = false;
      }
    }
    // // 检查是否在等待圆弧插补启动
    else if (this->circular_waiting_start) {
      // 检查是否已到达起始点
      if (abs(this->x_real - this->x_interpolation_start) <= POSITION_ERROR_THRESHOLD &&
          abs(this->y_real - this->y_interpolation_start) <= POSITION_ERROR_THRESHOLD) {
        // 已到达起始点，切换到圆弧插补模式
        this->mode = PLATFORM_MODE_CIRCULAR_INTERPOLATION;
        this->x->SetMode(x_linear_module::MODULE_MODE_POSITION);
        this->y->SetMode(x_linear_module::MODULE_MODE_POSITION);
        this->x_target = this->x_interpolation_final;
        this->y_target = this->y_interpolation_final;
        this->circular_waiting_start = false;
      }
    }

  }
  else if (this->mode == PLATFORM_MODE_FIND_HOME) {
    if (abs(this->x_real) <= POSITION_ERROR_THRESHOLD &&
        abs(this->y_real) <= POSITION_ERROR_THRESHOLD&&
        this->x->mode == x_linear_module::MODULE_MODE_POSITION &&
        this->y->mode == x_linear_module::MODULE_MODE_POSITION) {
      this->mode = PLATFORM_MODE_IDLE;
      this->homed = true;
      this->error_code = PLATFORM_ERROR_NONE;
    }
  } 
  else if (this->mode == PLATFORM_MODE_LINEAR_INTERPOLATION) {
    // 检查是否到达目标位置
    if (abs(this->x_real - this->x_target) <= POSITION_ERROR_THRESHOLD &&
        abs(this->y_real - this->y_target) <= POSITION_ERROR_THRESHOLD) {
      this->mode =PLATFORM_MODE_IDLE;
    }
    // //对水平/垂直直线做专门处理，避免 f==0 时插补轴不推进
    else if (abs(this->y_target - this->y_real)<= POSITION_ERROR_THRESHOLD) {
       // 水平线：只推进 X，Y 保持常量
      if (abs(this->x_target - this->x_interpolation_target) <= this->inter_step) {
        this->x_interpolation_target = this->x_target;
      } else {
        this->x_interpolation_target = this->x_real +((this->x_target >= this->x_interpolation_start) ? this->inter_step : -this->inter_step);
      }
    }
    else if (abs(this->x_target - this->x_real)<= POSITION_ERROR_THRESHOLD) {
       // 垂直线：只推进 Y，X 保持常量
      if (abs(this->y_target - this->y_interpolation_target) <= this->inter_step) {
        this->y_interpolation_target = this->y_target;
      } else {
        this->y_interpolation_target = this->y_real+((this->y_target >= this->y_interpolation_start) ? this->inter_step : -this->inter_step);
      }
    }
     //计算插补目标位置
    else if (this->x_target - this->x_real >0 &&this->y_target - this->y_real > 0) {
      // 第一象限
      if (LinearInterJudge(this->x_real, this->y_real, this->x_target,this->y_target, this->x_interpolation_start,this->y_interpolation_start) >= 0) {
        // 是否可以一步完成
        if (abs(this->x_target - this->x_interpolation_target) <=this->inter_step) {
          this->x_interpolation_target = this->x_target;
        } else {
          this->x_interpolation_target = this->x_real + this->inter_step;
        }
      } else {
        // 是否可以一步完成
        if (abs(this->y_target - this->y_interpolation_target) <=this->inter_step) {
          this->y_interpolation_target = this->y_target;
        } else {
          this->y_interpolation_target = this->y_real + this->inter_step;
        }
      }
    }
		        // 第二象限
        else if (this->x_target - this->x_real <= 0 && this->y_target - this->y_real > 0) {
            if (LinearInterJudge(this->x_real, this->y_real, this->x_target, this->y_target,
                                 this->x_interpolation_start, this->y_interpolation_start) >= 0) {
                if (abs(this->y_target - this->y_interpolation_target) <= this->inter_step)
                    this->y_interpolation_target = this->y_target;
                else
                    this->y_interpolation_target = this->y_real + this->inter_step;
            } else {
                if (abs(this->x_target - this->x_interpolation_target) <= this->inter_step)
                    this->x_interpolation_target = this->x_target;
                else
                    this->x_interpolation_target = this->x_real - this->inter_step;
            }
        }
        // 第三象限
        else if (this->x_target - this->x_real <= 0 && this->y_target - this->y_real <= 0) {
            if (LinearInterJudge(this->x_real, this->y_real, this->x_target, this->y_target,
                                 this->x_interpolation_start, this->y_interpolation_start) >= 0) {
                if (abs(this->x_target - this->x_interpolation_target) <= this->inter_step)
                    this->x_interpolation_target = this->x_target;
                else
                    this->x_interpolation_target = this->x_real - this->inter_step;
            } else {
                if (abs(this->y_target - this->y_interpolation_target) <= this->inter_step)
                    this->y_interpolation_target = this->y_target;
                else
                    this->y_interpolation_target = this->y_real - this->inter_step;
            }
        }
        // 第四象限
        else if (this->x_target - this->x_real > 0 && this->y_target - this->y_real <= 0) {
            if (LinearInterJudge(this->x_real, this->y_real, this->x_target, this->y_target,
                                 this->x_interpolation_start, this->y_interpolation_start) >= 0) {
                if (abs(this->y_target - this->y_interpolation_target) <= this->inter_step)
                    this->y_interpolation_target = this->y_target;
                else
                    this->y_interpolation_target = this->y_real - this->inter_step;
            } else {
                if (abs(this->x_target - this->x_interpolation_target) <= this->inter_step)
                    this->x_interpolation_target = this->x_target;
                else
                    this->x_interpolation_target = this->x_real + this->inter_step;
            }
        }
  /*  else if () {
      // 第二象限

    } else if () {
      // 第三象限
      
    } else if () {
      // 第四象限
      
    }*/
    // 设定插补目标位置
    this->x->SetTargetPositionWithVelocity(this->x_interpolation_target,this->inter_vel);
    this->y->SetTargetPositionWithVelocity(this->y_interpolation_target,this->inter_vel);
  } 
  else if (this->mode == PLATFORM_MODE_CIRCULAR_INTERPOLATION) {
     // 检查是否到达目标位置
    if (abs(this->x_real - this->x_target) <= POSITION_ERROR_THRESHOLD &&abs(this->y_real - this->y_target) <= POSITION_ERROR_THRESHOLD) {
      this->mode = PLATFORM_MODE_IDLE;
    }   
    else if (abs(this->x_target - this->x_interpolation_target) <=this->inter_step&&abs(this->y_target - this->y_interpolation_target) <=this->inter_step)
    {
      this->x_interpolation_target = this->x_target;
      this->y_interpolation_target = this->y_target;
    }
    else if ( this->x_real-this->x_center > 0 &&this->y_real-this->y_center >=0) {
      // 第一象限
      if (CircularInterJudge(this->x_real, this->y_real, this->x_center,this->y_center, this->radius) >= 0) 
      {
        if (this->clockwise) {
          this->y_interpolation_target = this->y_real - this->inter_step;
        } 
        else {
          this->x_interpolation_target = this->x_real - this->inter_step; 
        }
      } 
      else {
        if (this->clockwise) {
          this->x_interpolation_target = this->x_real + this->inter_step;
        } 
        else {
          this->y_interpolation_target = this->y_real + this->inter_step;
        }
      }
    }
    else if ( this->x_real-this->x_center <= 0 &&this->y_real-this->y_center > 0) {
      // 第二象限
      if (CircularInterJudge(this->x_real, this->y_real, this->x_center,this->y_center, this->radius) >= 0) 
      {
        if (this->clockwise) {
          this->x_interpolation_target = this->x_real + this->inter_step;
        } 
        else {
          this->y_interpolation_target = this->y_real - this->inter_step;
        }
      } 
      else {
        if (this->clockwise) {
          this->y_interpolation_target = this->y_real + this->inter_step;
        } 
        else {
            this->x_interpolation_target = this->x_real - this->inter_step;
        }
      }
    }
		        else if (this->x_real - this->x_center <= 0 && this->y_real - this->y_center <= 0) {
            // 第三象限
            if (CircularInterJudge(this->x_real, this->y_real, this->x_center, this->y_center, this->radius) >= 0) {
                if (this->clockwise) this->y_interpolation_target = this->y_real + this->inter_step;
                else this->x_interpolation_target = this->x_real + this->inter_step;
            } else {
                if (this->clockwise) this->x_interpolation_target = this->x_real - this->inter_step;
                else this->y_interpolation_target = this->y_real - this->inter_step;
            }
        }
        else if (this->x_real - this->x_center >= 0 && this->y_real - this->y_center <= 0) {
            // 第四象限
            if (CircularInterJudge(this->x_real, this->y_real, this->x_center, this->y_center, this->radius) >= 0) {
                if (this->clockwise) this->x_interpolation_target = this->x_real - this->inter_step;
                else this->y_interpolation_target = this->y_real + this->inter_step;
            } else {
                if (this->clockwise) this->y_interpolation_target = this->y_real - this->inter_step;
                else this->x_interpolation_target = this->x_real + this->inter_step;
            }
        }
   /* else if ( ) {
      // 第三象限
      
    }
    else if () {
      // 第四象限
      
    }*/
    this->x->SetTargetPositionWithVelocity(this->x_interpolation_target,this->inter_vel);
    this->y->SetTargetPositionWithVelocity(this->y_interpolation_target,this->inter_vel);
  }
  else if (this->mode == PLATFORM_MODE_CLOSED_LOOP) {
    // 判断是否到达目标位置
    if (abs(this->x_real - this->x_target) <= POSITION_ERROR_THRESHOLD &&
        abs(this->y_real - this->y_target) <= POSITION_ERROR_THRESHOLD) {
      this->x->SetTargetVelocity(0);
      this->y->SetTargetVelocity(0);
      this->mode = PLATFORM_MODE_IDLE;
      // 清零PID积分项
      this->pos_pid_x.integral = 0;
      this->pos_pid_y.integral = 0;
      return;
    }
    // 计算x和y的目标速度
    float vel_x = pos_pid_x.Calc(this->x_real);
    float vel_y = pos_pid_y.Calc(this->y_real);
    // 设置x和y的目标速度
    this->x->SetTargetVelocity(vel_x);
    this->y->SetTargetVelocity(vel_y);
  }

  // 请完成此函数 End
}

void XYplatform::Stop(void) {
  this->mode = PLATFORM_MODE_IDLE;
  this->x->SetTargetVelocityHard(0.0f);
  this->y->SetTargetVelocityHard(0.0f);
  this->x->SetMode(x_linear_module::MODULE_MODE_IDLE);
  this->y->SetMode(x_linear_module::MODULE_MODE_IDLE);
  this->pos_pid_x.integral = 0;
  this->pos_pid_y.integral = 0;
}

void XYplatform::GetStatus(float *curr_x, float *curr_y, uint8_t *status) {
  if (curr_x != nullptr) {
    *curr_x = this->x->GetPosition();
    // *curr_x = this->x_real;
  }
  if (curr_y != nullptr) {
    *curr_y = this->y->GetPosition();
    // *curr_y = this->y_real;
  }

  if (status != nullptr) {
    if (this->error_code != PLATFORM_ERROR_NONE ||
        this->x->mode == x_linear_module::MODULE_MODE_ERROR ||
        this->y->mode == x_linear_module::MODULE_MODE_ERROR) {
      *status = 0xFF;
    } else if (this->mode == PLATFORM_MODE_FIND_HOME) {
      *status = 0x01;
    } else if (this->mode == PLATFORM_MODE_LINEAR_INTERPOLATION ||this->mode == PLATFORM_MODE_CIRCULAR_INTERPOLATION ||this->mode == PLATFORM_MODE_CLOSED_LOOP) {
      *status = 0x02;
    } 
    else if(this->mode==PLATFORM_MODE_MANUAL)
    {
      *status=0x03;
    }
    else {
      *status = 0x00;
    }
  }
}
} // namespace xy_platform
