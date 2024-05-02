def parse_contents(contents):
    global checkboxes
    content_type, content_string = contents.split(',')
    #global checkboxes
    decoded = base64.b64decode(content_string)
    global df  # Hacer referencia a la variable global df
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    options = [{'label': col, 'value': col} for col in df.columns]

    return options
@app.callback([
               Output('column-selector', 'options'),
               
               ],
              [Input('upload-data', 'contents')])
def update_output(contents):
    if contents is None:
        raise PreventUpdate
    else:
        options = parse_contents(contents)
       
        return [options]
