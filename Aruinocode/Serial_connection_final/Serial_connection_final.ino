char buffer[10]; // 9 digits + 1 null terminator
void setup() {
  // 31Hz
  //TCCR0A = 0b00000001; // 
  //TCCR0B = 0b00000101; // 
    Serial.begin(115200);
}

void loop() {
    int sensorValue = analogRead(A0);
       Serial.println(sensorValue);
    delay(60);
    
  
    if (Serial.available() > 0) {
        int num_bytes = Serial.readBytesUntil('\n', buffer, sizeof(buffer)-1);
        buffer[num_bytes] = '\0';
        if (num_bytes == 9) {
            char buf1[4] = {buffer[0], buffer[1], buffer[2], '\0'}; // first 3 digits
            char buf2[4] = {buffer[3], buffer[4], buffer[5], '\0'}; // next 3 digits
            char buf3[4] = {buffer[6], buffer[7], buffer[8], '\0'}; // last 3 digits
            
            if (isdigit(buf1[0]) && isdigit(buf2[0]) && isdigit(buf3[0])) {
                int val1 = atoi(buf1);
                int val2 = atoi(buf2);
                int val3 = atoi(buf3);
              // Do something with the values
                /*
                Serial.print("Values: ");
                Serial.print(val1);
                Serial.print(", ");
                Serial.print(val2);
                Serial.print(", ");
                Serial.println(val3);
                */
                analogWrite(6, val1);
                analogWrite(9, val2);
                analogWrite(10, val3);
            } else {
                Serial.println("Invalid input");
            }
        } else {
            Serial.println("Invalid input");
        }
    }
}
