using System;
using OpenTK.Graphics.OpenGL;
using OpenTK.Mathematics;
using OpenTK.Windowing.Common;
using OpenTK.Windowing.Desktop;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

class Program
{
    static void Main()
    {
        var gameSettings = GameWindowSettings.Default;
        var windowSettings = new NativeWindowSettings()
        {
            ClientSize = new Vector2i(800, 600),
            Title = "Cube Rotation Relative to Camera"
        };

        using (var game = new Game(gameSettings, windowSettings))
        {
            game.Run();
        }
    }
}

class Game : GameWindow
{
    private int _texture;
    private int _shaderProgram;
    private int _vao;
    private int _vbo;
    private int _ebo;

    private Vector3 _cameraPosition = new Vector3(0.0f, 0.0f, 5.0f);
    private Vector3 _cameraTarget = Vector3.Zero;

    private float _rotationAngle = 0.0f;
    private Vector3 _rotationAxis = Vector3.UnitY;

    private bool _rotateAroundCameraTarget = true;

    public Game(GameWindowSettings gameSettings, NativeWindowSettings windowSettings)
        : base(gameSettings, windowSettings) { }

    protected override void OnLoad()
    {
        base.OnLoad();

        GL.ClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        GL.Enable(EnableCap.DepthTest);

        // Load shaders
        _shaderProgram = CreateShaderProgram();
        GL.UseProgram(_shaderProgram);

        // Load texture
        _texture = LoadTexture("texture.png");

        // Define a 3D cube with texture coordinates
        float[] vertices = {
            // Positions          // Texture Coords
            -1.0f, -1.0f, -1.0f,  0.0f, 0.0f,
             1.0f, -1.0f, -1.0f,  1.0f, 0.0f,
             1.0f,  1.0f, -1.0f,  1.0f, 1.0f,
            -1.0f,  1.0f, -1.0f,  0.0f, 1.0f,
            -1.0f, -1.0f,  1.0f,  0.0f, 0.0f,
             1.0f, -1.0f,  1.0f,  1.0f, 0.0f,
             1.0f,  1.0f,  1.0f,  1.0f, 1.0f,
            -1.0f,  1.0f,  1.0f,  0.0f, 1.0f
        };

        uint[] indices = {
            0, 1, 2, 2, 3, 0, // Back face
            4, 5, 6, 6, 7, 4, // Front face
            0, 4, 7, 7, 3, 0, // Left face
            1, 5, 6, 6, 2, 1, // Right face
            3, 2, 6, 6, 7, 3, // Top face
            0, 1, 5, 5, 4, 0  // Bottom face
        };

        // Set up VAO, VBO, and EBO
        _vao = GL.GenVertexArray();
        GL.BindVertexArray(_vao);

        _vbo = GL.GenBuffer();
        GL.BindBuffer(BufferTarget.ArrayBuffer, _vbo);
        GL.BufferData(BufferTarget.ArrayBuffer, vertices.Length * sizeof(float), vertices, BufferUsageHint.StaticDraw);

        _ebo = GL.GenBuffer();
        GL.BindBuffer(BufferTarget.ElementArrayBuffer, _ebo);
        GL.BufferData(BufferTarget.ElementArrayBuffer, indices.Length * sizeof(uint), indices, BufferUsageHint.StaticDraw);

        int positionLocation = GL.GetAttribLocation(_shaderProgram, "aPosition");
        GL.VertexAttribPointer(positionLocation, 3, VertexAttribPointerType.Float, false, 5 * sizeof(float), 0);
        GL.EnableVertexAttribArray(positionLocation);

        int texCoordLocation = GL.GetAttribLocation(_shaderProgram, "aTexCoord");
        GL.VertexAttribPointer(texCoordLocation, 2, VertexAttribPointerType.Float, false, 5 * sizeof(float), 3 * sizeof(float));
        GL.EnableVertexAttribArray(texCoordLocation);
    }

    protected override void OnRenderFrame(FrameEventArgs args)
    {
        base.OnRenderFrame(args);

        GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);

        GL.UseProgram(_shaderProgram);

        Matrix4 view = Matrix4.LookAt(_cameraPosition, _cameraTarget, Vector3.UnitY);
        Matrix4 projection = Matrix4.CreatePerspectiveFieldOfView(MathHelper.DegreesToRadians(45.0f), Size.X / (float)Size.Y, 0.1f, 100.0f);

        // Choose rotation center based on mode
        Matrix4 model;
        if (_rotateAroundCameraTarget)
        {
            model = Matrix4.CreateTranslation(-_cameraTarget) *
                    Matrix4.CreateFromAxisAngle(_rotationAxis, MathHelper.DegreesToRadians(_rotationAngle)) *
                    Matrix4.CreateTranslation(_cameraTarget);
        }
        else
        {
            model = Matrix4.CreateFromAxisAngle(_rotationAxis, MathHelper.DegreesToRadians(_rotationAngle));
        }

        Matrix4 mvp = model * view * projection;

        int mvpLocation = GL.GetUniformLocation(_shaderProgram, "uMVP");
        GL.UniformMatrix4(mvpLocation, false, ref mvp);

        // Bind texture
        GL.ActiveTexture(TextureUnit.Texture0);
        GL.BindTexture(TextureTarget.Texture2D, _texture);

        GL.BindVertexArray(_vao);
        GL.DrawElements(PrimitiveType.Triangles, 36, DrawElementsType.UnsignedInt, 0);

        SwapBuffers();
    }

    protected override void OnUpdateFrame(FrameEventArgs args)
    {
        base.OnUpdateFrame(args);

        if (KeyboardState.IsKeyDown(OpenTK.Windowing.GraphicsLibraryFramework.Keys.Left))
        {
            _rotationAxis = Vector3.UnitY;
            _rotationAngle -= 50.0f * (float)args.Time;
        }

        if (KeyboardState.IsKeyDown(OpenTK.Windowing.GraphicsLibraryFramework.Keys.Right))
        {
            _rotationAxis = Vector3.UnitY;
            _rotationAngle += 50.0f * (float)args.Time;
        }

        if (KeyboardState.IsKeyDown(OpenTK.Windowing.GraphicsLibraryFramework.Keys.Up))
        {
            _rotationAxis = Vector3.UnitX;
            _rotationAngle += 50.0f * (float)args.Time;
        }

        if (KeyboardState.IsKeyDown(OpenTK.Windowing.GraphicsLibraryFramework.Keys.Down))
        {
            _rotationAxis = Vector3.UnitX;
            _rotationAngle -= 50.0f * (float)args.Time;
        }

        if (KeyboardState.IsKeyDown(OpenTK.Windowing.GraphicsLibraryFramework.Keys.Space))
        {
            _rotateAroundCameraTarget = !_rotateAroundCameraTarget;
        }
    }

    protected override void OnUnload()
    {
        base.OnUnload();

        GL.DeleteTexture(_texture);
        GL.DeleteProgram(_shaderProgram);
        GL.DeleteVertexArray(_vao);
        GL.DeleteBuffer(_vbo);
        GL.DeleteBuffer(_ebo);
    }

    private int CreateShaderProgram()
    {
        const string vertexShaderSource = @"#version 330 core
        layout (location = 0) in vec3 aPosition;
        layout (location = 1) in vec2 aTexCoord;

        uniform mat4 uMVP;

        out vec2 TexCoord;

        void main()
        {
            gl_Position = uMVP * vec4(aPosition, 1.0);
            TexCoord = aTexCoord;
        }";

        const string fragmentShaderSource = @"#version 330 core
        in vec2 TexCoord;
        out vec4 FragColor;

        uniform sampler2D uTexture;

        void main()
        {
            FragColor = texture(uTexture, TexCoord);
        }";

        int vertexShader = CompileShader(ShaderType.VertexShader, vertexShaderSource);
        int fragmentShader = CompileShader(ShaderType.FragmentShader, fragmentShaderSource);

        int program = GL.CreateProgram();
        GL.AttachShader(program, vertexShader);
        GL.AttachShader(program, fragmentShader);
        GL.LinkProgram(program);

        GL.DeleteShader(vertexShader);
        GL.DeleteShader(fragmentShader);

        return program;
    }

    private int CompileShader(ShaderType type, string source)
    {
        int shader = GL.CreateShader(type);
        GL.ShaderSource(shader, source);
        GL.CompileShader(shader);

        GL.GetShader(shader, ShaderParameter.CompileStatus, out int status);
        if (status == 0)
        {
            string info = GL.GetShaderInfoLog(shader);
            throw new Exception($"Shader compilation failed: {info}");
        }

        return shader;
    }

    private int LoadTexture(string path)
    {
        using var image = Image.Load<Rgba32>(path);

        // Flip the image vertically
        var flippedImage = image.Clone(ctx => ctx.Flip(FlipMode.Vertical));

        // Convert the image to a byte array
        var pixelData = new byte[4 * flippedImage.Width * flippedImage.Height];
        flippedImage.CopyPixelDataTo(pixelData);

        int texture = GL.GenTexture();
        GL.BindTexture(TextureTarget.Texture2D, texture);

        GL.TexImage2D(TextureTarget.Texture2D, 0, PixelInternalFormat.Rgba,
            flippedImage.Width, flippedImage.Height, 0,
            OpenTK.Graphics.OpenGL.PixelFormat.Rgba, PixelType.UnsignedByte, pixelData);

        GL.GenerateMipmap(GenerateMipmapTarget.Texture2D);

        return texture;
    }
}
