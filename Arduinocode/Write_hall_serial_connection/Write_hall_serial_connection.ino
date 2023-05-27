void setup() {
  Serial.begin(115200);

  

}

void loop() {
  int Hall_left1 = analogRead(A0);
  int Hall_left2 = analogRead(A1);
  int Hall_right1 = analogRead(A2);
  int Hall_right2 = analogRead(A3);
  int Hall_back1 = analogRead(A4);
  int Hall_back2 = analogRead(A5);
  //Serial.println(Hall_left1);
  //Serial.println(Hall_left2);
  Serial.println();
  Serial.println(Hall_right1);
  Serial.println(Hall_right2);
  //Serial.println(Hall_back1);
  //Serial.println(Hall_back2);
  delay(500);

}
