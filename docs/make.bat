@ECHO OFF

set SOURCEDIR=.
set BUILDDIR=_build

if "%SPHINXBUILD%"=="" (
	set SPHINXBUILD=sphinx-build
)

if "%1"=="clean" (
	if exist %BUILDDIR% rmdir /S /Q %BUILDDIR%
	goto end
)

%SPHINXBUILD% -b html %SOURCEDIR% %BUILDDIR%/html

:end