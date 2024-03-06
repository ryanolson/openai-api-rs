use serde::de::{self, SeqAccess, Visitor};
use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;
use std::collections::HashMap;
use std::fmt;
use utoipa::ToSchema;

use crate::impl_builder_methods;
use crate::v1::common;

#[derive(ToSchema, Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub enum ToolChoiceType {
    None,
    Auto,
    ToolChoice { tool: Tool },
}

#[derive(ToSchema, Debug, Serialize, Deserialize, Clone)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatCompletionMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<HashMap<String, i32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(serialize_with = "serialize_tool_choice")]
    pub tool_choice: Option<ToolChoiceType>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ChatCompletionMessage {
    pub role: MessageRole,

    #[serde(deserialize_with = "deserialize_content")]
    pub content: Content,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl ChatCompletionRequest {
    pub fn new(model: String, messages: Vec<ChatCompletionMessage>) -> Self {
        Self {
            model,
            messages,
            temperature: None,
            top_p: None,
            stream: None,
            n: None,
            response_format: None,
            stop: None,
            max_tokens: None,
            presence_penalty: None,
            frequency_penalty: None,
            logit_bias: None,
            user: None,
            seed: None,
            tools: None,
            tool_choice: None,
        }
    }
}

impl_builder_methods!(
    ChatCompletionRequest,
    temperature: f32,
    top_p: f32,
    n: i32,
    response_format: Value,
    stream: bool,
    stop: Vec<String>,
    max_tokens: i32,
    presence_penalty: f32,
    frequency_penalty: f32,
    logit_bias: HashMap<String, i32>,
    user: String,
    seed: i64,
    tools: Vec<Tool>,
    tool_choice: ToolChoiceType
);

#[derive(ToSchema, Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum MessageRole {
    user,
    system,
    assistant,
    function,
}

#[derive(ToSchema, Debug, Deserialize, Clone, PartialEq, Eq)]
pub enum Content {
    Text(String),
    ImageUrl(Vec<ImageUrl>),
}

impl serde::Serialize for Content {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match *self {
            Content::Text(ref text) => serializer.serialize_str(text),
            Content::ImageUrl(ref image_url) => image_url.serialize(serializer),
        }
    }
}

fn deserialize_content<'de, D>(deserializer: D) -> Result<Content, D::Error>
where
    D: Deserializer<'de>,
{
    struct ContentVisitor;

    impl<'de> Visitor<'de> for ContentVisitor {
        type Value = Content;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("a string or an array of content parts")
        }

        fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
        where
            E: de::Error,
        {
            Ok(Content::Text(value.to_owned()))
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: SeqAccess<'de>,
        {
            let mut parts = Vec::new();
            while let Some(value) = seq.next_element::<String>()? {
                if value.starts_with("http://") || value.starts_with("https://") {
                    parts.push(ImageUrl {
                        r#type: ContentType::image_url,
                        text: None,
                        image_url: Some(ImageUrlType { url: value }),
                    });
                } else {
                    parts.push(ImageUrl {
                        r#type: ContentType::text,
                        text: Some(value),
                        image_url: None,
                    });
                }
            }
            Ok(Content::ImageUrl(parts))
        }
    }

    deserializer.deserialize_any(ContentVisitor)
}

#[derive(ToSchema, Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum ContentType {
    text,
    image_url,
}

#[derive(ToSchema, Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub struct ImageUrlType {
    pub url: String,
}

#[derive(ToSchema, Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub struct ImageUrl {
    pub r#type: ContentType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_url: Option<ImageUrlType>,
}

/// A full chat completion.
pub type ChatCompletionResponse = ChatCompletionGeneric<ChatCompletionChoice>;

/// A delta chat completion, which is streamed token by token.
pub type ChatCompletionResponseDelta = ChatCompletionGeneric<ChatCompletionChoiceDelta>;

#[derive(Serialize, Deserialize, Debug)]
pub struct ChatCompletionGeneric<C>
where
    C: Serialize,
{
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<C>,
    pub usage: Option<common::Usage>,
    pub system_fingerprint: Option<String>,
}

// todo add logprobs
#[derive(Deserialize, Serialize, Debug)]
pub struct ChatCompletionChoice {
    pub index: u64,
    pub finish_reason: FinishReason,
    pub message: ChatCompletionContent,
}

// todo add logprobs
/// Same as ChatCompletionMessage, but received during a response stream.
#[derive(ToSchema, Debug, Serialize, Deserialize)]
pub struct ChatCompletionChoiceDelta {
    pub index: u64,
    pub finish_reason: Option<FinishReason>,
    pub delta: ChatCompletionContent,
}

#[derive(ToSchema, Debug, Deserialize, Serialize)]
pub struct ChatCompletionContent {
    pub role: MessageRole,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(ToSchema, Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
pub struct Function {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: FunctionParameters,
}

#[derive(ToSchema, Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum JSONSchemaType {
    Object,
    Number,
    String,
    Array,
    Null,
    Boolean,
}

#[derive(ToSchema, Debug, Deserialize, Serialize, Clone, Default, PartialEq, Eq)]
pub struct JSONSchemaDefine {
    #[serde(rename = "type")]
    pub schema_type: Option<JSONSchemaType>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enum_values: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<HashMap<String, Box<JSONSchemaDefine>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub items: Option<Box<JSONSchemaDefine>>,
}

#[derive(ToSchema, Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
pub struct FunctionParameters {
    #[serde(rename = "type")]
    pub schema_type: JSONSchemaType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<HashMap<String, Box<JSONSchemaDefine>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<Vec<String>>,
}

#[derive(ToSchema, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum FinishReason {
    stop,
    length,
    content_filter,
    tool_calls,
    null,
}

#[derive(ToSchema, Debug, Deserialize, Serialize)]
#[allow(non_camel_case_types)]
pub struct FinishDetails {
    pub r#type: FinishReason,
    pub stop: String,
}

#[derive(ToSchema, Debug, Deserialize, Serialize, Clone)]
pub struct ToolCall {
    pub id: String,
    pub r#type: String,
    pub function: ToolCallFunction,
}

#[derive(ToSchema, Debug, Deserialize, Serialize, Clone)]
pub struct ToolCallFunction {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

fn serialize_tool_choice<S>(
    value: &Option<ToolChoiceType>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    match value {
        Some(ToolChoiceType::None) => serializer.serialize_str("none"),
        Some(ToolChoiceType::Auto) => serializer.serialize_str("auto"),
        Some(ToolChoiceType::ToolChoice { tool }) => {
            let mut map = serializer.serialize_map(Some(2))?;
            map.serialize_entry("type", &tool.r#type)?;
            map.serialize_entry("function", &tool.function)?;
            map.end()
        }
        None => serializer.serialize_none(),
    }
}

#[derive(ToSchema, Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
pub struct Tool {
    pub r#type: ToolType,
    pub function: Function,
}

#[derive(ToSchema, Debug, Deserialize, Serialize, Copy, Clone, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ToolType {
    Function,
}
