
const int SENSOR_1_1 = A0;
const int SENSOR_1_2 = A1;
const int SENSOR_1_3 = A2;
const int SENSOR_2_1 = A3;
const int SENSOR_2_2 = A4;
const int SENSOR_2_3 = A5;
const int SENSOR_3_1 = A6;
const int SENSOR_3_2 = A7;
const int SENSOR_3_3 = A8;
const float ADC_TO_VOLTAGE_CONVERTOR = 4.88;
const float VOLTAGE_STEP = 10.0;
const int READ_DELAY = 100;
const int PROCESS_DELAY = 2000;


float calculateMedian(float value1, float value2, float value3)
{
  if ((value1 >= value2 && value1 <= value3) || (value1 <= value2 && value1 >= value3))
  {
    return value1;
  } 
  else if ((value2 >= value1 && value2 <= value3) || (value2 <= value1 && value2 >= value3))
  {
    return value2;
  } 
  else 
  {
    return value3;
  }
}

float readValue(int pin)
{
  // float Values[3] = {0.0, 0.0, 0.0};
  float value = 0.0;
  
  int inputValue = analogRead(pin);
  float voltage = (inputValue * ADC_TO_VOLTAGE_CONVERTOR);
  value = voltage / VOLTAGE_STEP;
  return value;
}

void setup() {
  // put your setup code here, to run once:

  Serial.begin(9600);
}

void loop() {
  // put your main code here, to run repeatedly:

  float TMR1 = calculateMedian(readValue(SENSOR_1_1),readValue(SENSOR_1_2),readValue(SENSOR_1_3));
  Serial.println("Read pin 1 Sensor 1 value: " + String(SENSOR_1_1));
  Serial.println("Read pin 2 Sensor 1 value: " + String(SENSOR_1_2));
  Serial.println("Read pin 3 Sensor 1 value: " + String(SENSOR_1_3));
  Serial.println("The result of the first layer TMR: " + String(TMR1) + "\n");
  
  float TMR2 = calculateMedian(readValue(SENSOR_2_1),readValue(SENSOR_2_2),readValue(SENSOR_2_3));
  Serial.println("Read pin 1 Sensor 2 value: " + String(SENSOR_2_1));
  Serial.println("Read pin 2 Sensor 2 value: " + String(SENSOR_2_2));
  Serial.println("Read pin 3 Sensor 2 value: " + String(SENSOR_2_3));
  Serial.println("The result of the first layer TMR: " + String(TMR2) + "\n");
  
  float TMR3 = calculateMedian(readValue(SENSOR_3_1),readValue(SENSOR_3_2),readValue(SENSOR_3_3));
  Serial.println("Read pin 1 Sensor 3 value: " + String(SENSOR_3_1));
  Serial.println("Read pin 2 Sensor 3 value: " + String(SENSOR_3_2));
  Serial.println("Read pin 3 Sensor 3 value: " + String(SENSOR_3_3));
  Serial.println("The result of the first layer TMR: " + String(TMR3) + "\n");

  float finalTemperature = calculateMedian(TMR1, TMR2, TMR3);
  Serial.println("**** The result of the second layer's TMR (Final Temperature): " + String(finalTemperature) + " °C ****\n");

  delay(PROCESS_DELAY);
}