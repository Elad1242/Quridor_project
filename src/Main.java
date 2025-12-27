import javafx.application.Application;
import javafx.stage.Stage;
import ui.GameController;

/**
 * Entry point for the Quoridor game.
 * Requires Java 17+ and JavaFX SDK 17+.
 */
public class Main extends Application {

    @Override
    public void start(Stage stage) {
        new GameController(stage);
        stage.show();
    }

    public static void main(String[] args) {
        launch(args);
    }
}
