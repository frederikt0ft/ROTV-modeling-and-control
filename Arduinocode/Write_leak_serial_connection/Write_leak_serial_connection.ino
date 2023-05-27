void setup() {
    Serial.begin(115200);
}

void loop() {
  int Leak_sensor = digitalRead(D7);

  Serial.println(Leak_sensor);

  delay(500)


  
}
