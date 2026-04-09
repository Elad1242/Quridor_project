@echo off
REM Launch the Quoridor GUI with Human vs MLBot option
set JAVA_HOME=C:\Users\eladt\.jdks\ms-17.0.17
set PATH=%JAVA_HOME%\bin;%PATH%
set JAVAFX=C:\Users\eladt\Downloads\openjfx-17.0.17_windows-x64_bin-sdk\javafx-sdk-17.0.17\lib

java --module-path "%JAVAFX%" --add-modules javafx.controls,javafx.graphics,javafx.fxml -cp out\production\Quridor_project Main
