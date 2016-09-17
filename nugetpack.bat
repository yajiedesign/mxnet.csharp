call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" amd64
cd mxnet.csharp
msbuild mxnet.csharp.csproj /t:Build /p:Configuration="Release 4.0"
msbuild mxnet.csharp.csproj /t:Build /p:Configuration="Release 4.5"
msbuild mxnet.csharp.csproj /t:Build /p:Configuration="Release 4.6"
msbuild mxnet.csharp.csproj /t:Build /p:Configuration="Release 4.6.1"
nuget pack mxnet.csharp.csproj -Prop Configuration=Release;Platform=x64 -OutputDirectory ..\nugetpublish
pause