char buffer[4];

void setup() {



  // 31Hz
  TCCR0A = 0b00000001; // 
  TCCR0B = 0b00000101; // 

  // 62 Hz
  TCCR0A = 0b00000011; // 
  TCCR0B = 0b00000101; //

  // 125Hz
  TCCR0A = 0b00000001; // 
  TCCR0B = 0b00000100; //

  // 250 Hz
  TCCR0A = 0b00000011; // 
  TCCR0B = 0b00000100; //

  // 500Hz
  TCCR0A = 0b00000001; // 
  TCCR0B = 0b00000011; // 

  // 1000Hz
  TCCR0A = 0b00000011; // 
  TCCR0B = 0b00000011; // 

  // 4000Hz
  TCCR0A = 0b00000001; // 
  TCCR0B = 0b00000010; // 

  // 31Hz
  TCCR0A = 0b00000001; // 
  TCCR0B = 0b00000101; // 




    Serial.begin(115200);
}



void loop() {
      if (Serial.available() > 0) {
        int num_bytes = Serial.readBytesUntil('\n', buffer, sizeof(buffer)-1);
        buffer[num_bytes] = '\0';
        if (isdigit(buffer[0])) {
            int num = atoi(buffer);
            // Do something with the integer num
            Serial.println(num);
            analogWrite(6, num);
        } else {
            Serial.println("Invalid input");
        }
    }
   
   
   
}
