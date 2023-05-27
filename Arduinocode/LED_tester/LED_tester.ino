void setup() {
  pinMode(4 , OUTPUT);
  // 125Hz
  //TCCR0A = 0b00000001; // 
  //TCCR0B = 0b00000100; //


    Serial.begin(115200);
    


}

void loop() {
  digitalWrite(4,HIGH);
  delay(1000);
  digitalWrite(4,LOW);
  delay(1000);
  Serial.print("virker");

} 
