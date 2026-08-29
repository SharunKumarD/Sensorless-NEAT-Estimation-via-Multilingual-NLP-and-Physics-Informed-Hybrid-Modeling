import React, { useState } from 'react';
import { View, Text, TextInput, TouchableOpacity, StyleSheet, SafeAreaView, KeyboardAvoidingView, Platform, Alert, ScrollView, ActivityIndicator } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import { Audio } from 'expo-av';

const API_BASE_URL = 'http://192.168.1.6:8000';

export default function DataInputScreen() {
  const [recording, setRecording] = useState<Audio.Recording | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [isTextProcessing, setIsTextProcessing] = useState(false);
  const [isAudioProcessing, setIsAudioProcessing] = useState(false);
  const [inputText, setInputText] = useState('');
  const [weight, setWeight] = useState('69');
  const [intensity, setIntensity] = useState('Normal'); 
  const navigation = useNavigation<any>();

  const isAnyProcessing = isTextProcessing || isAudioProcessing;

  const handleRecordPress = async () => {
    try {
      if (isRecording) {
        setIsRecording(false);
        if (recording) {
          await recording.stopAndUnloadAsync();
          const uri = recording.getURI();
          console.log('Recording stopped and stored at', uri);
          setRecording(null);
          
          if (!uri) return;

          setIsAudioProcessing(true);
          
          const formData = new FormData();
          
          // @ts-ignore
          formData.append('file', {
            uri: Platform.OS === 'android' ? uri : uri.replace('file://', ''),
            name: 'recording.m4a',
            type: 'audio/m4a'
          } as any);
          formData.append('weight', weight.toString());
          formData.append('intensity', intensity);
          
          try {
            const response = await fetch(`${API_BASE_URL}/process-audio`, {
              method: 'POST',
              body: formData,
              headers: {
                'Accept': 'application/json',
              },
            });
            
            if (response.status === 429) {
              Alert.alert('Server Busy', 'The AI is taking a quick breather. Please wait 15 seconds and try again!');
              return;
            }

            const json = await response.json();
            
            if (json.status === 'success') {
              navigation.navigate('Dashboard', { dashboardData: json.data });
            } else {
              const errMsg = String(json.message || '');
              if (errMsg.includes('429') || errMsg.includes('RESOURCE_EXHAUSTED')) {
                Alert.alert('Server Busy', 'The AI is taking a quick breather. Please wait 15 seconds and try again!');
              } else {
                Alert.alert('Error', json.message || 'Failed to process audio.');
              }
            }
          } catch (fetchError) {
            console.error(fetchError);
            Alert.alert('Network Error', 'Could not connect to the backend server.');
          } finally {
            setIsAudioProcessing(false);
          }
        }
      } else {
        const permission = await Audio.requestPermissionsAsync();
        if (permission.status === 'granted') {
          await Audio.setAudioModeAsync({
            allowsRecordingIOS: true,
            playsInSilentModeIOS: true,
          });
          const { recording: newRecording } = await Audio.Recording.createAsync(
            Audio.RecordingOptionsPresets.HIGH_QUALITY
          );
          setRecording(newRecording);
          setIsRecording(true);
        } else {
          Alert.alert('Permission Required', 'Please grant microphone permissions.');
        }
      }
    } catch (err) {
      console.error('Failed to start/stop recording', err);
      setIsRecording(false);
      setIsAudioProcessing(false);
    }
  };

  const parseDuration = (text: string) => {
    let durationMins = 10;
    const hrRegex = /(\d+(?:\.\d+)?)\s*(?:hr|hour)s?/i;
    const minRegex = /(\d+(?:\.\d+)?)\s*(?:min|m)s?/i;
    
    const hrMatch = text.match(hrRegex);
    const minMatch = text.match(minRegex);
    
    if (hrMatch) {
      durationMins = parseFloat(hrMatch[1]) * 60;
    } else if (minMatch) {
      durationMins = parseFloat(minMatch[1]);
    }
    
    return durationMins;
  };

  const handleCalculate = async () => {
    try {
      setIsTextProcessing(true);
      const duration_mins = parseDuration(inputText);
      const weight_kg = parseFloat(weight) || 69.0;
      
      const payload = {
        weight_kg: weight_kg,
        met_value: 3.5,
        temperature: 24.0,
        humidity: 60.0,
        duration_mins: duration_mins,
        acoustic_biomarkers: [[0.1, 1500, 0.05]],
        current_fatigue_level: 40.0,
        intensity: intensity,
        transcript: inputText
      };

      const response = await fetch(`${API_BASE_URL}/calculate_neat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      });

      if (response.status === 429) {
        Alert.alert('Server Busy', 'The AI is taking a quick breather. Please wait 15 seconds and try again!');
        return;
      }

      const json = await response.json();

      if (json.status === 'success') {
        navigation.navigate('Dashboard', { dashboardData: json.data });
      } else {
        const errMsg = String(json.message || '');
        if (errMsg.includes('429') || errMsg.includes('RESOURCE_EXHAUSTED')) {
          Alert.alert('Server Busy', 'The AI is taking a quick breather. Please wait 15 seconds and try again!');
        } else {
          Alert.alert('Error', json.message || 'Failed to calculate NEAT parameters.');
        }
      }
    } catch (error) {
      console.error(error);
      Alert.alert('Network Error', `Could not connect to the backend server at ${API_BASE_URL}.`);
    } finally {
      setIsTextProcessing(false);
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <KeyboardAvoidingView 
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.inner}
      >
        <ScrollView contentContainerStyle={styles.scrollContainer} keyboardShouldPersistTaps="handled">
          <View style={styles.headerContainer}>
            <Text style={styles.title}>NEAT Tracker</Text>
            <Text style={styles.subtitle}>Stage 1: Data Input</Text>
          </View>

          <View style={styles.microphoneContainer}>
            <TouchableOpacity 
              style={[styles.micButton, isRecording && styles.micButtonRecording]} 
              onPress={handleRecordPress}
              activeOpacity={0.8}
              disabled={isAnyProcessing}
            >
              {isAudioProcessing ? (
                <ActivityIndicator size="large" color="#fff" />
              ) : (
                <Ionicons name="mic" size={64} color={isRecording ? "#fff" : "#007BFF"} />
              )}
            </TouchableOpacity>
            <Text style={styles.recordingText}>
              {isAudioProcessing ? "Analyzing Audio..." : (isRecording ? "Recording... Tap to stop" : "Tap to start recording")}
            </Text>
          </View>

          <View style={styles.inputContainer}>
            <Text style={styles.inputLabel}>Weight (kg)</Text>
            <TextInput
              style={styles.singleLineInput}
              keyboardType="numeric"
              placeholder="e.g. 69"
              placeholderTextColor="#999"
              value={weight}
              onChangeText={setWeight}
              editable={!isAnyProcessing}
            />

            <Text style={[styles.inputLabel, { marginTop: 16 }]}>Intensity Level</Text>
            <View style={styles.intensityContainer}>
              {['Low', 'Normal', 'High'].map((level) => (
                <TouchableOpacity
                  key={level}
                  style={[
                    styles.intensityButton,
                    intensity === level && styles.intensityButtonActive
                  ]}
                  onPress={() => setIntensity(level)}
                  disabled={isAnyProcessing}
                >
                  <Text style={[
                    styles.intensityButtonText,
                    intensity === level && styles.intensityButtonTextActive
                  ]}>
                    {level}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>

            <Text style={[styles.inputLabel, { marginTop: 16 }]}>Manual Input / Transcript</Text>
            <TextInput
              style={styles.textInput}
              multiline
              placeholder="Enter text (e.g., 'Naan indha route-la 5km nadandhen for 30min')"
              placeholderTextColor="#999"
              value={inputText}
              onChangeText={setInputText}
              editable={!isAnyProcessing}
            />
            <TouchableOpacity 
              style={[styles.submitButton, isTextProcessing && styles.submitButtonDisabled]} 
              onPress={handleCalculate}
              disabled={isAnyProcessing}
            >
              {isTextProcessing ? (
                <ActivityIndicator color="#fff" />
              ) : (
                <Text style={styles.submitButtonText}>Submit Data & Calculate</Text>
              )}
            </TouchableOpacity>
          </View>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F9FF',
  },
  inner: {
    flex: 1,
  },
  scrollContainer: {
    flexGrow: 1,
    padding: 24,
    justifyContent: 'space-between',
  },
  headerContainer: {
    marginTop: 20,
    alignItems: 'center',
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#333',
  },
  subtitle: {
    fontSize: 16,
    color: '#666',
    marginTop: 8,
  },
  microphoneContainer: {
    alignItems: 'center',
    justifyContent: 'center',
    flex: 1,
    paddingVertical: 30,
  },
  micButton: {
    width: 150,
    height: 150,
    borderRadius: 75,
    backgroundColor: '#E6F0FA',
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#007BFF',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 10,
    elevation: 8,
  },
  micButtonRecording: {
    backgroundColor: '#FF3B30',
    shadowColor: '#FF3B30',
  },
  recordingText: {
    marginTop: 20,
    fontSize: 16,
    color: '#555',
    fontWeight: '500',
  },
  inputContainer: {
    width: '100%',
  },
  inputLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: '#333',
    marginBottom: 8,
  },
  singleLineInput: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    fontSize: 16,
    borderWidth: 1,
    borderColor: '#E0E0E0',
    color: '#333',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 3,
    elevation: 2,
  },
  intensityContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  intensityButton: {
    flex: 1,
    backgroundColor: '#fff',
    borderWidth: 1,
    borderColor: '#E0E0E0',
    paddingVertical: 12,
    borderRadius: 8,
    alignItems: 'center',
    marginHorizontal: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 2,
    elevation: 1,
  },
  intensityButtonActive: {
    backgroundColor: '#007BFF',
    borderColor: '#007BFF',
  },
  intensityButtonText: {
    fontSize: 14,
    color: '#666',
    fontWeight: '500',
  },
  intensityButtonTextActive: {
    color: '#fff',
    fontWeight: 'bold',
  },
  textInput: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    minHeight: 120, // Allows it to be multi-line while setting a baseline height
    textAlignVertical: 'top',
    fontSize: 16,
    borderWidth: 1,
    borderColor: '#E0E0E0',
    color: '#333',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 3,
    elevation: 2,
  },
  submitButton: {
    backgroundColor: '#007BFF',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
    marginTop: 24,
    marginBottom: 20, // Padding at bottom for keyboard clearance
  },
  submitButtonDisabled: {
    backgroundColor: '#999',
  },
  submitButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
});
