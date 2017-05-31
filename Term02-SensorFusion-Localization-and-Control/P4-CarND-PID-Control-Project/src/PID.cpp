#include "PID.h"

//using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {}

void PID::Init(double _Kp, double _Ki, double _Kd) {
	/*
  * Errors
  */
  p_error = 0.0;
  i_error = 0.0;
  d_error = 0.0;

  /*
  * Coefficients
  */ 
  Kp = _Kp;
  Ki = _Ki;
  Kd = _Kd;
}

void PID::UpdateError(double cte) {
	
	if((cte - p_error) != 0)
	{
		d_error = cte - p_error;
		p_error = cte;
		i_error += cte;
	}

}

double PID::TotalError() {
	// Return the total error of each coefficient multiplied by the respective error
	return -Kp * p_error - Kd * d_error - Ki * i_error;
}

