

def write_pascal_voc(file_path, filename, width, height, xtl, ytl, xbr, ybr):
    with open(file_path, 'w') as file:
        xml_content = f'''<annotation>
          <folder>frame</folder>
          <filename>{filename}</filename>
          <source>
            <database>Unknown</database>
            <annotation>Unknown</annotation>
            <image>Unknown</image>
          </source>
          <size>
            <width>{width}</width>
            <height>{height}</height>
            <depth></depth>
          </size>
          <segmented>0</segmented>
          <object>
            <name>queen</name>
            <truncated>0</truncated>
            <occluded>0</occluded>
            <difficult>0</difficult>
            <bndbox>
              <xmin>{xtl}</xmin>
              <ymin>{ytl}</ymin>
              <xmax>{xbr}</xmax>
              <ymax>{ybr}</ymax>
            </bndbox>
            <attributes>
              <attribute>
                <name>rotation</name>
                <value>0.0</value>
              </attribute>
              <attribute>
                <name>track_id</name>
                <value>0</value>
              </attribute>
              <attribute>
                <name>keyframe</name>
                <value>True</value>
              </attribute>
            </attributes>
          </object>
        </annotation>'''
        file.write(xml_content)
