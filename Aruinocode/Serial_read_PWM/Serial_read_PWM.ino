
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



    Serial.begin(115200);
}

int num = 127;

void loop() {
   analogWrite(6, num);
   
}
