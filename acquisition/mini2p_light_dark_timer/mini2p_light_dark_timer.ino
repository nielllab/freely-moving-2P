// Mini2p recording chamber light control
// Connect digital pin #4 to board controlling chamber lights,
// this will toggle lights on/off over a given pair of intervals.
// DMM, May 2025
// last modified June 18 2025

unsigned long interval = 60*3; // units of seconds


unsigned long previousMillis = 0;
unsigned long currentMillis = 0;
bool pinState = HIGH;
int pin = 4;

void setup() {
  pinMode(pin, OUTPUT);
  digitalWrite(pin, pinState);
}

void loop() {
  currentMillis = millis()/1000;

  if ((currentMillis - previousMillis) >= interval) {
    previousMillis = currentMillis;
    pinState = !pinState;
    digitalWrite(pin, pinState);
  };

}
