@echo off
REM ============================================================================
REM  Quoridor GUI launcher
REM ----------------------------------------------------------------------------
REM  Uses the machine's current Java 17 (Adoptium) and the locally installed
REM  JavaFX SDK at %USERPROFILE%\Downloads\javafx-sdk-17.0.17\lib.
REM
REM  Classes are read from out\production\Quridor_project, which is the
REM  IntelliJ default build output. If this directory is empty, build the
REM  project from IntelliJ once (Build -> Build Project, or Ctrl+F9) or run
REM  the COMPILE block below by invoking this script with the -build flag.
REM ============================================================================

set JAVA_HOME=C:\Program Files\Eclipse Adoptium\jdk-17.0.16.8-hotspot
set PATH=%JAVA_HOME%\bin;%PATH%
set JAVAFX=%USERPROFILE%\Downloads\javafx-sdk-17.0.17\lib
set OUT=out\production\Quridor_project

if "%1"=="-build" (
    echo [run_gui] Compiling project to %OUT% ...
    if not exist "%OUT%" mkdir "%OUT%"
    dir /s /b src\*.java > .sources.tmp
    javac -encoding UTF-8 -d "%OUT%" -sourcepath src ^
        --module-path "%JAVAFX%" ^
        --add-modules javafx.base,javafx.controls,javafx.graphics,javafx.fxml ^
        @.sources.tmp
    set COMPILE_ERR=%ERRORLEVEL%
    del .sources.tmp
    if not "%COMPILE_ERR%"=="0" (
        echo [run_gui] Compilation failed.
        exit /b %COMPILE_ERR%
    )
    echo [run_gui] Compiled successfully.
)

if not exist "%OUT%\Main.class" (
    echo [run_gui] ERROR: %OUT% has no compiled classes.
    echo [run_gui] Either build the project in IntelliJ ^(Ctrl+F9^),
    echo [run_gui] or re-run this script with the -build flag:
    echo [run_gui]     run_gui.bat -build
    exit /b 1
)

java --module-path "%JAVAFX%" ^
     --add-modules javafx.controls,javafx.graphics,javafx.fxml ^
     -cp "%OUT%" Main
