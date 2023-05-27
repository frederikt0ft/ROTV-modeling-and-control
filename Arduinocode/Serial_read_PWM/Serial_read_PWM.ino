char buffer[13]; // 9 digits + 1 null terminator








void setup() {


  pinMode(4, OUTPUT);   
  pinMode(7, OUTPUT);   
  pinMode(8, OUTPUT);   


  // 31Hz
  //TCCR0A = 0b00000001; // 
  //TCCR0B = 0b00000101; // 

  // 62 Hz
  //TCCR0A = 0b00000011; // 
  //TCCR0B = 0b00000101; //

  // 125Hz
  TCCR0A = 0b00000001; // 
  TCCR0B = 0b00000100; //

  // 250 Hz
  //TCCR0A = 0b00000011; // 
  //TCCR0B = 0b00000100; //

  // 500Hz
  //TCCR0A = 0b00000001; // 
  //TCCR0B = 0b00000011; // 

  // 1000Hz
  //TCCR0A = 0b00000011; // 
  //TCCR0B = 0b00000011; // 

  // 4000Hz
  //TCCR0A = 0b00000001; // 
  //TCCR0B = 0b00000010; // 

  // 31Hz
  //TCCR0A = 0b00000001; //   
  //TCCR0B = 0b00000101; // 
    Serial.begin(115200);
}

void loop() {
    if (Serial.available() > 0) {
        int num_bytes = Serial.readBytesUntil('\n', buffer, sizeof(buffer)-1);
        buffer[num_bytes] = '\0';
        if (num_bytes == 12) {
            char buf1[4] = {buffer[0], buffer[1], buffer[2], '\0'}; // first 3 digits
            char buf2[4] = {buffer[3], buffer[4], buffer[5], '\0'}; // next 3 digits
            char buf3[4] = {buffer[6], buffer[7], buffer[8], '\0'}; // last 3 digits
            char buf4[4] = {buffer[9], buffer[10], buffer[11], '\0'}; // first 3 digits

            
            if (isdigit(buf1[0]) && isdigit(buf2[0]) && isdigit(buf3[0]) && isdigit(buf4[0])) {
                int val1 = atoi(buf1);
                int val2 = atoi(buf2);
                int val3 = atoi(buf3);
                int val4 = atoi(buf4);

                                // Split the integer into individual digits
                int dir1 = val4 / 100; // Extract the first digit
                int dir2 = (val4 / 10) % 10; // Extract the second digit
                int dir3 = val4 % 10; // Extract the third digit


                // Do something with the values
                /*
                Serial.print("Values: ");
                Serial.print(val1);
                Serial.print(", ");
                Serial.print(val2);
                Serial.print(", ");
                Serial.print(val3);
                Serial.print(", "); 
                Serial.print(dir1);
                Serial.print(dir2);
                Serial.println(dir3);
                */


                analogWrite(6, val1);          //Left PWM
                analogWrite(9, val2);          //Right PWM
                analogWrite(10, val3);         //Tail PWM

   
                digitalWrite(4,dir1);
                digitalWrite(7,dir2);     //Right DIR
                digitalWrite(8,dir3);     //Tail DIR

            }/* else {
                Serial.println("Invalid input");
            */}/*
        } else {
            Serial.println("Invalid input");
        }*/
    }
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
