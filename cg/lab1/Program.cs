using System;
using SFML.Graphics;
using SFML.System;
using SFML.Window;

class Program
{
    static void Main(string[] args)
    {
        const int windowWidth = 800;
        const int windowHeight = 600;

        // Создаем окно
        RenderWindow window = new RenderWindow(new VideoMode(windowWidth, windowHeight), "Bresenham Circle Algorithm");
        window.Closed += (sender, e) => window.Close();

        int centerX = windowWidth / 2;
        int centerY = windowHeight / 2;
        int radius = 100;
        int radiusChange = 1;
        bool increasing = true;

        Clock clock = new Clock();

        while (window.IsOpen)
        {
            window.DispatchEvents();

            // Обновление радиуса для анимации пульсации
            if (clock.ElapsedTime.AsMilliseconds() > 10)
            {
                if (increasing)
                {
                    radius += radiusChange;
                    if (radius >= 200) increasing = false;
                }
                else
                {
                    radius -= radiusChange;
                    if (radius <= 50) increasing = true;
                }
                clock.Restart();
            }

            // Очистка экрана
            window.Clear(Color.Black);

            // Отрисовка окружности
            DrawCircle(window, centerX, centerY, radius);

            // Отображение содержимого окна
            window.Display();
        }
    }

    static void DrawCircle(RenderWindow window, int centerX, int centerY, int radius)
    {
        int x = 0;
        int y = radius;
        int d = 3 - 2 * radius;

        while (x <= y)
        {
            DrawSymmetricPoints(window, centerX, centerY, x, y);

            if (d < 0)
            {
                d += 4 * x + 6;
            }
            else
            {
                d += 4 * (x - y) + 10;
                y--;
            }
            x++;
        }
    }

    static void DrawSymmetricPoints(RenderWindow window, int centerX, int centerY, int x, int y)
    {
        PutPixel(window, centerX + x, centerY + y);
        PutPixel(window, centerX - x, centerY + y);
        PutPixel(window, centerX + x, centerY - y);
        PutPixel(window, centerX - x, centerY - y);
        PutPixel(window, centerX + y, centerY + x);
        PutPixel(window, centerX - y, centerY + x);
        PutPixel(window, centerX + y, centerY - x);
        PutPixel(window, centerX - y, centerY - x);
    }

    static void PutPixel(RenderWindow window, int x, int y)
    {
        if (x >= 0 && x < window.Size.X && y >= 0 && y < window.Size.Y)
        {
            CircleShape pixel = new CircleShape(1) { FillColor = Color.White };
            pixel.Position = new Vector2f(x, y);
            window.Draw(pixel);
        }
    }
}
