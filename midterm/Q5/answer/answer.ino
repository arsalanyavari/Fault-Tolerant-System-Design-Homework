
const int SENSOR_1 = A0;
const int SENSOR_2 = A1;
const int SENSOR_3 = A2;
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
  float Values[3] = {0.0, 0.0, 0.0};
  float value = 0.0;
  
  for (int i = 0; i < 3; i++)
  {
    int inputValue = analogRead(pin);
    float voltage = (inputValue * ADC_TO_VOLTAGE_CONVERTOR);
    value = voltage / VOLTAGE_STEP;
    Values[i] = value;
    Serial.println("Read " + String(i) + " value: " + String(value));
    delay(READ_DELAY);
  }

  float TMR_values = calculateMedian(Values[0], Values[1], Values[2]);
  Serial.println("The result of the first layer TMR: " + String(TMR_values) + "\n");
  return TMR_values;
}

void setup() {
  // put your setup code here, to run once:

  Serial.begin(9600);
}

void loop() {
  // put your main code here, to run repeatedly:

  float temperature[3] = {0.0, 0.0, 0.0};

  temperature[0] = readValue(SENSOR_1);
  temperature[1] = readValue(SENSOR_2);
  temperature[2] = readValue(SENSOR_3);

  float finalTemperature = calculateMedian(temperature[0], temperature[1], temperature[2]);

  Serial.println("**** The result of the second layer's TMR (Final Temperature): " + String(finalTemperature) + " °C ****\n");

  delay(PROCESS_DELAY);
}
