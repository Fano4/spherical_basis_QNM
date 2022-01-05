python3 -m pip install --upgrade pip
python3 -m pip install --upgrade build
python3 -m pip install --upgrade twine
reinstall=False
UPLOAD=False

if [ -d "build" ]; then
reinstall=True
rm -r dist/
fi
python3 -m build
if [ ${reinstall} ]; then
  python3 -m pip install --force-reinstall --no-deps ./dist/*.tar.gz
else
  python3 -m pip install --no-deps ./dist/*.tar.gz
fi


POSITIONAL=()
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    -u|--upload)
      UPLOAD=True
      shift # past argument
      shift # past value
      ;;
    *)    # unknown option
      POSITIONAL+=("$1") # save it in an array for later
      shift # past argument
      ;;
  esac
done

set -- "${POSITIONAL[@]}" # restore positional parameters

# if [ $UPLOAD ]; then
# python3 -m twine upload --verbose --repository testpypi dist/*
# fi

